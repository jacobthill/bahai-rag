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
df_precision = pd.read_csv('data/NB - Precision - L2.csv', dtype={'Citation #': str})
df_grounded = pd.read_csv('data/NB - L3_L5.csv', dtype={'Citation #': str})

df_precision['Citation #'] = df_precision['Citation #'].str.replace('NB-', '', regex=False)

df_grounded = df_grounded[df_grounded['Citation #'].str.isnumeric() == True]

full_answer = """
The famine that swept through Iraq and surrounding regions from 1879 to 1881 (1296–1298 A.H.) was a period of "extreme hardship" for the Bahá’í community, particularly for those exiled in Mosul (Hadbá’) [1][2]. Bahá’u’lláh acknowledged that "famine has overtaken most lands this year," noting that His followers were among those "oppressed and downtrodden" [3][4].

The Condition of the Community in Mosul
During this crisis, many Bahá’ís were held as captives in Mosul, having been banished from Baghdad [5][6]. Under the leadership of Jinab-i-Zayn, the community displayed "astonishing patience" and "amazing trust" in God, which moved local observers to wonder at these "strangers in their midst who were neither distressed nor terrified" but continued to "offer praise day and night" [5][6]. In the face of starvation, Bahá’u’lláh testified that "their souls ate of food from Heaven" while their bodies endured "infinite hardships and sufferings" [5][7].

Bahá’u’lláh’s Direct Response
Bahá’u’lláh’s response was characterized by both spiritual counsel and practical financial relief. He urged the believers to "conduct themselves with the utmost wisdom" and cautioned against the public spreading of news regarding their plight without permission, as "the spreading of such matters by the friends may cause certain incidents" with the authorities [3][8].

To alleviate the suffering, He dispatched funds through trusted messengers like Shaykh Salmán [1]. For instance, He sent a sum of fifty tumans in Rabi'u'th-Thani 1298 A.H. (March 1881) [1]. He explicitly commanded that these sums reach Zaynu'l-Muqarrabín in Mosul so that he could "distribute it among the captives of that land" [9].

The Use of Huqúqu’lláh Funds
The sources confirm that Huqúqu’lláh (the Right of God) was explicitly utilized to respond to this famine [2][10].
A Shift in Policy: Bahá’u’lláh stated that while "demanding [Huqúqu’lláh] is not and hath never been beloved," the "apparent conditions" and "extraordinary expenses" of the time necessitated a firm decree for its collection [10][11].
Mandate for Collection: He instructed His representatives to "collect [Huqúqu’lláh] and send it to Jinab-i-Amin or Jinab-i-Samandar" because "in these days, some in various regions have fallen into great hardship, including the families of the prisoners who are both in debt and face delays in their daily sustenance" [2].
Specific Allocations: Bahá’u’lláh provided detailed instructions on how these funds were to be dispersed:
Widows and Orphans: Funds were allocated for "the children of those who were martyred" and "handmaidens who remain without spouse" [12][13].
Prisoner Relief: Specific sums were assigned to the families of prominent exiles and prisoners, such as Jináb-i-Dhabíh (50 tumans), the family of Ustád Ahmad (10 tumans), and the family of Karbílá’í Taqí (15 tumans) [14]
The Poor: In some instances, He directed that half of the collected Huqúq be "given to the poor of that land" directly [15]
Inter-Community Aid: Bahá’u’lláh praised those who "secretly sent sustenance" to the oppressed, noting that even some "followers of the Spirit" (Christians) had aided the Bahá’ís out of "pure compassion" during these lean years [16].

Ultimately, Bahá’u’lláh regarded the provision of these funds as a means of "preservation and purity of wealth and of blessing," ensuring that the "divine rights" would prevent the friends from experiencing ultimate "hardship" [10][17]. Sentiments of "boundless joy" were obtained when these relief efforts succeeded in helping the community "remain firm like a mighty mountain" against the "tempests of ordeal" [18][19].
"""

merged_context = df_grounded.merge(df_precision, on='Citation #', how='inner')
unique_contexts = merged_context['Citation'].unique().tolist()

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
