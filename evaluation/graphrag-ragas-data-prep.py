import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# Use Langchain objects directly - they are much more stable for setting parameters
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# Import the metric CLASSES so we can initialize them explicitly
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate

# 2. Initialize LLM and Embeddings via LangChain
# This ensures max_tokens is handled correctly without "unexpected keyword" errors
lc_llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=4096,
    temperature=0
)
lc_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 3. Wrap them for RAGAS
# This bridge is the official way to ensure Ragas metrics can talk to the LLM
eval_llm = LangchainLLMWrapper(lc_llm)
eval_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

# 4. Initialize Metrics explicitly
# This solves the "All metrics must be initialised metric objects" error
# and the "AttributeError: embed_query" error.
metric_faithfulness = Faithfulness(llm=eval_llm)
metric_relevancy = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)

# --- Data Loading ---
df_precision = pd.read_csv('data/GR - Precision - L2.csv', dtype={'Citation #': str})
df_grounded = pd.read_csv('data/GR - L3_L5.csv', dtype={'Citation #': str})

df_precision['Citation #'] = df_precision['Citation #'].str.replace('GR-', '', regex=False)

full_answer = """
Based on the provided historical documents, here is a detailed summary regarding the response of Bahá’u’lláh and the Bahá’í community to the famines in Iraq (specifically Mosul/Hadba) around 1879–1880.
### Summary of Events
**1. Acknowledgment of the Crisis and Call for Discretion**
During the severe famine that overtook "most lands," particularly affecting Mosul (referred to as *Hadba*), Bahá’u’lláh issued specific instructions to the Bahá’í community. While acknowledging the hardship, He emphasized the necessity of **wisdom (hikmat)** and **secrecy**. He advised that publicizing the community's internal affairs or their relief efforts could lead to dangerous "incidents" or persecution. The believers were counseled to reflect on the Quranic verses regarding tests and patience during these tribulations.
**2. The Mobilization of Funds**
The community actively mobilized financial resources to assist those in the affected areas. Funds were collected in various regions, including Baghdad (referred to as *Ardh-i-Salam* or *Ardh-i-Ta*) and arguably Hamadan (*Ardh-i-Ha and Mim*), to be sent to Mosul. Specific sums, such as thirty tumans, are mentioned as being ready for transfer to Mosul to alleviate the distress.
**3. Utilization of Huquq’u’llah and Charitable Contributions**
Huquq’u’llah funds were indeed utilized, but under strict protocols established by Bahá’u’lláh to ensure safety and order.
* **Voluntary Contribution:** Bahá’u’lláh stipulated that payments must be made with "perfect desire and longing."
* **Trusteeship:** Funds were to be given only to trusted individuals (trustees/Amin).
* **Documentation:** Receipts (*qabz*) were mandatory for these transactions.
* **Divine Permission:** The expenditure and management of these funds required specific permission to ensure they were handled according to Divine pleasure.
Additionally, general charity toward the poor and wayfarers was highly praised by Bahá’u’lláh as a "beloved" act that brings down blessings.
***
### Detailed Evidence and Citations
#### 1. The Circular Instruction on Secrecy and Patience
Bahá’u’lláh sent a specific instruction regarding the situation in Mosul to **Mirza Abu'l-Hasan**, which was subsequently quoted in letters to other believers (such as **Samandar** and **Varqa**) to ensure the policy was understood. He linked the famine to divine testing but warned that spreading news of the believers' activities could cause trouble.
#### 2. Protocol for Huquq’u’llah and Financial Contributions
Bahá’u’lláh gave explicit permission for believers to pay Huquq’u’llah to assist in these matters, provided strict administrative procedures (receipts and trustees) were followed to maintain order and security.
#### 3. Transfer of Funds to Mosul (Hadba)
The texts confirm that funds were collected in other cities to be sent to the famine-stricken area of Mosul.
#### 4. Praise for General Charity
Beyond organized funds, Bahá’u’lláh expressed deep satisfaction with believers who personally supported the poor during these times.
"""

unique_contexts = df_precision['Citation'].dropna().unique().tolist()

ragas_data = {
    "question": ["Give me a detailed summary of the response of Bahá’u’lláh and the Bahá’í community to the widespread famines in Iraq from 1879 onwards. Were Huquq’u’llah funds used? If so, how?"],
    "answer": [full_answer],
    "contexts": [unique_contexts]
}

dataset = Dataset.from_dict(ragas_data)
print(f"Dataset created with {len(unique_contexts)} source contexts.")

# 5. Run the Evaluation
# Using the pre-initialized metric objects
result = evaluate(
    dataset,
    metrics=[metric_faithfulness, metric_relevancy]
)

# 6. Export and Print
results_df = result.to_pandas()
results_df.to_csv('data/RAGAS_Evaluation_Results.csv', index=False)
print("\nEvaluation Complete. Scores:")
print(results_df[['faithfulness', 'answer_relevancy']])
