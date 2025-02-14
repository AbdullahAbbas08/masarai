from http.client import responses

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_ollama import OllamaLLM
from collections import defaultdict
import json

from transformers import AutoModel
import torch
import faiss

import pandas as pd
import numpy as np






embedding_model = None
index_user = None
data_user = None
index_work = None
data_work = None

def load_model_and_data():
    global embedding_model, index_user, data_user,index_work, data_work

    if embedding_model is None:
        # Load the embedding model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        embedding_model = AutoModel.from_pretrained(r"E:\ForEveryOne\__AbdullahAbbas\ai\masaraiv2\.venv\embedding_model_and_data\jina-embeddings-v3",trust_remote_code=True).to(device)

        # Load the data
        data_work = pd.read_excel(r'E:\ForEveryOne\__AbdullahAbbas\ai\masaraiv2\.venv\embedding_model_and_data\PSU.xlsx')

        # Load the FAISS index
        file_path = r'E:\ForEveryOne\__AbdullahAbbas\ai\masaraiv2\.venv\embedding_model_and_data\PSU.npy'
        embeddings = np.load(file_path)
        faiss.normalize_L2(embeddings)

        # Create the FAISS index
        embedding_dim = embeddings.shape[1]
        index_work = faiss.IndexFlatIP(embedding_dim)
        index_work.add(embeddings)
        data_user = pd.read_excel(r'E:\ForEveryOne\__AbdullahAbbas\ai\masaraiv2\.venv\embedding_model_and_data\users.xlsx')

        # Load the FAISS index
        file_path = r'E:\ForEveryOne\__AbdullahAbbas\ai\masaraiv2\.venv\embedding_model_and_data\user_embeddings.npy'
        embeddings = np.load(file_path)
        faiss.normalize_L2(embeddings)

        # Create the FAISS index
        embedding_dim = embeddings.shape[1]
        index_user = faiss.IndexFlatIP(embedding_dim)
        index_user.add(embeddings)


        print("Model and data loaded successfully!")
    else:
        print("Model and data already loaded. Skipping...")

def get_to_from_cc(all_text,description,DelegationType='Organization'):
  response_schemas = [
      ResponseSchema(name="from", description="المسمى الوظيفي أو اسم المرسل"),
      ResponseSchema(name="to", description="قائمة المسمى الوظيفي أو أسماء الأشخاص المستلمين "),
      ResponseSchema(name="cc", description="قائمة المسمى الوظيفي أو أسماء الأشخاص الذين تم إرسال نسخة لهم"),
  ]

  # Create the output parser
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

  # Get the format instructions for the prompt
  format_instructions = output_parser.get_format_instructions()

  prompt_template = PromptTemplate(
      input_variables=["all_text", "description","DelegationType"],
      template="""
      مهمتك هي تحليل محتوى البريد الإلكتروني واستخراج المعلومات التالية بدقة عالية بناءً على نوع التفويض (DelegationType):

      **القواعد الأساسية:**
      1. استخرج الوظائف فقط عند توفرها (مع المكان إذا ذُكر)
      2. تجاهل الأسماء إلا إذا لم تتوفر الوظيفة
      3. افحص النص والوصف معاً للحصول على المعلومات
      4. تأكد من عدم تكرار نفس المعلومات

      **المؤشرات المهمة:**
      - **FROM:**
        - ابحث عن أي إشارة للمرسل في بداية البريد
        - تحقق من التوقيع في نهاية البريد
        - راجع الوصف (description) للمعلومات الإضافية عن المرسل

      - **TO:**
        - في حالة **DelegationType = Organization**، استخرج فقط **المنظمات** المذكورة
        - في حالة **DelegationType = Employee**، استخرج فقط **الموظفين**
        - الكلمات الدالة على المنظمات مثل: [ "وكالة", "عمادة", "مركز", "أكاديمية", "وزارة", "هيئة", "نائب", "وحدة", "مجلس", "مكتب", "كلية", "قسم", "مساعد", "لجنة", "المكرم", "الجهات التالية"]
        - إذا بدأ البريد بـ "تعميم" فهو موجه للجهة المذكورة بعده
        - تجاهل أي إشارات تبدأ بكلمات مثل "نسخة" أو "صورة"

      - **CC:**
        - في حالة **DelegationType = Organization**، استخرج فقط **المنظمات** ذات الصلة
        - الكلمات الدالة: ["نسخة", "صور", "لمدير", "لسعادة"]
        - تأكد من أن هذه المعلومات لم تُذكر في حقل TO

      - **EmployeeCC:**
        - يحتوي على جميع الموظفين المستخرجين وفقًا لنوع التفويض
        - الكلمات الدالة على كنية الموظفين مثل: ["دكتور", "أستاذ", "رئيس", "سلمه", "السادة", "سعادة","معالي", "سمو"]
        - في حالة **DelegationType = Organization**، استخرج **الموظفين** داخل المنظمة وضعهم هنا
        - في حالة **DelegationType = Employee**، استخرج **الموظفين** وضعهم هنا

      **قواعد المعالجة:**
      - تحقق من السياق الكامل قبل استخراج أي معلومة
      - افصل بين TO و CC بناءً على الكلمات الدالة ونوع التفويض
      - لا تكرر نفس المعلومة في أكثر من حقل
      - استخدم الوصف لتأكيد أو إكمال المعلومات الناقصة

      **النص المراد تحليله:**
      {all_text}

      **الوصف الإضافي:**
      {description}

      {format_instructions}

      **DelegationType:**
      {DelegationType}
      """
  )


  llm = OllamaLLM(model="qwen2.5:14b-instruct-fp16", temperature=0)

  # Create the LLMChain
  chain =  prompt_template | llm

  # Input email text
  # Run the chain
  response = chain.invoke({"DelegationType":DelegationType,"all_text": all_text,"description":description, "format_instructions": format_instructions})
  print('response : ',response)
  # Parse the output using the output parser
  parsed_output = output_parser.parse(response)
  parsed_output = {
      "to": parsed_output["to"],
      "cc": parsed_output["cc"],

  }
  with open("to_from_cc.json", "w", encoding="utf-8") as json_file:
    json.dump(parsed_output, json_file, ensure_ascii=False, indent=4)

  print("to_from_cc file saved successfully!")
  return parsed_output

def convert_to_standard_types(obj):
    """Recursively convert numpy types to standard Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_standard_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_standard_types(item) for item in obj]
    elif isinstance(obj, np.int64):  # Check for numpy int64
        return int(obj)  # Convert to standard int
    return obj

def get_organization(parsed_output):
    global embedding_model, index_user, data_user, index_work, data_work
    load_model_and_data()

    output = {
        "To": [],
        "CC": [],
        "EmployeeCC": [],
    }

    phrases_to_remove = ["فضيلة", "سعادة", "لسعادة", "سعادته", "سعادتة"]

    # Process both 'to' and 'cc'
    for key in ['to', 'cc']:
        for query in parsed_output.get(key, '').split(','):
            query = query.strip()

            # Remove specified phrases
            for phrase in phrases_to_remove:
                query = query.replace(phrase, '').strip()

            if not query:
                continue

            # Remove المكلف if last word
            if query.split()[-1] == "المكلف":
                query = ' '.join(query.split()[:-1])

            # Save cleaned query for potential employee search
            employee_query = query

            # Modify query for organization search
            if query.split(' ')[0] in ['المدير', 'مدير']:
                query = 'مكتب ' + query
            new_query = query + ' ' + query.split(' ')[-1]

            # Organization search
            query_embedding = embedding_model.encode([new_query], show_progress_bar=False, task='text-matching')
            faiss.normalize_L2(query_embedding)
            distances, indices = index_work.search(query_embedding, k=1)

            if distances[0][0] < 0.45:
                org_id = '0'
                org_name = query
            else:
                org_id = data_work['OrganizationId'][indices[0][0]]
                org_name = data_work['OrganizationNameAr'][indices[0][0]]

            org_entry = {"Id": org_id, "Name": org_name}

            if key == 'to':
                output['To'].append(org_entry)
            elif key == 'cc':
                if distances[0][0] > 0.45:
                    output['CC'].append(org_entry)
                else:
                    # Employee search
                    user_embedding = embedding_model.encode([employee_query], show_progress_bar=False,
                                                            task='text-matching')
                    faiss.normalize_L2(user_embedding)
                    user_distances, user_indices = index_user.search(user_embedding, k=1)

                    if user_distances[0][0] < 0.45:
                        emp_id = data_user['EmployeeId'][user_indices[0][0]]
                        emp_name = data_user['EmployeeNameAr'][user_indices[0][0]]
                        output['EmployeeCC'].append({"Id": emp_id, "Name": emp_name})
                    else:
                        output['EmployeeCC'].append({"Id": "0", "Name": employee_query})



    # Convert numpy types and handle empty lists
    output = convert_to_standard_types(output)
    # for key in output:
    #     if isinstance(output[key], list):
    #         if len(output[key]) == 0:
    #             output[key] = None
    #         elif len(output[key]) == 1:
    #             output[key] = output[key][0]

    # Save JSON
    with open("organization_name_id.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print("Organization file saved successfully!")
    return output