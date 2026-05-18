# AI Learning

來源：喵哩文創〈2026 AI 免費課程完整指南：30+ 官方資源一次整理〉  
主題：免費 AI 學習資源、平台比較、學習路徑與 100 小時學習規劃  
整理日期：2026-05-18

---

## 一、文章核心摘要

2026 年的 AI 免費學習資源已經非常充足，問題不再是「哪裡有課可以學」，而是「不同背景的人應該先學什麼」。文章整理了 DeepLearning.AI、Anthropic Academy、OpenAI Academy、Google AI、Microsoft GitHub、AWS Skill Builder、Hugging Face、Fast.ai、Kaggle、LangChain Academy、Stanford、MIT、李宏毅課程等平台，並從時間效率、市場認可度、可遷移性、動手程度與更新頻率等角度評估。

文章的主要判斷是：免費資源對大多數 AI 學習目標已經足夠。對初學者而言，應先建立概念基礎，再選擇開發、企業認證或工具應用其中一條主線深入，最後用實作驗證學習成果。

---

## 二、學習資源分類總覽

| 類型 | 代表平台 | 適合對象 | 核心價值 |
|---|---|---|---|
| 基礎概念 | DeepLearning.AI、Coursera 旁聽、AI for Everyone | 初學者、非技術背景、轉職者 | 建立 AI 與 ML 的概念地基 |
| 官方工具課 | OpenAI Academy、Anthropic Academy、Google AI | 想學 ChatGPT、Claude、Gemini 實務應用者 | 學官方產品與 API 的正確用法 |
| 工程實作 | Microsoft GitHub、Kaggle、Fast.ai | 工程師、具 Python 基礎者 | 直接動手跑程式、建立專案 |
| 雲端與認證 | AWS Skill Builder、Google Cloud Skills Boost | 企業 IT、雲端顧問、AI 導入角色 | 建立雲端 AI 服務與認證能力 |
| 開源與理論 | Hugging Face、Stanford CS224N、MIT 6.S191、李宏毅 ML | 想深入 LLM、NLP、深度學習者 | 理解模型原理與開源生態 |
| Agent / RAG 應用 | LangChain Academy、Anthropic MCP、Microsoft AI Agents | AI 應用開發者 | 學習 agent、RAG、MCP 等現代 AI 應用架構 |

---

## 三、DeepLearning.AI 與吳恩達課程

### 定位

吳恩達系課程被文章視為 AI 教育的基礎。其優勢不是最新，而是概念清晰、結構穩定，適合用來建立後續學習所需的共同語言。

### 建議課程

- **AI for Everyone**：非技術背景的最佳起點。
- **Machine Learning Specialization**：理解機器學習基本概念與方法。
- **Deep Learning Specialization**：進一步理解深度學習架構。
- **Generative AI with LLMs**：Coursera 與 AWS 聯合出品，聚焦 LLM。
- **AI Agentic Design Patterns**：聚焦 agent 架構。
- **MCP 短課**：快速理解 Model Context Protocol。
- **ChatGPT Prompt Engineering for Developers**：與 OpenAI 聯合出品。
- **RAG 系列短課**：聚焦向量資料庫與檢索增強生成。

### Coursera 旁聽方式

Coursera 上許多 DeepLearning.AI 課程可以免費旁聽。通常在課程頁面點選「Enroll for Free」後選擇「Audit」即可觀看影片與教材。旁聽通常不能提交作業，也不能取得付費認證。

### 重點結論

DeepLearning.AI 短課程被文章評為免費 AI 資源中投報率最高的一類。若只想先打基礎，建議順序是：

1. AI for Everyone
2. Machine Learning Specialization
3. DeepLearning.AI GenAI 短課
4. RAG、Prompt Engineering、MCP 或 Agent 相關短課

---

## 四、Anthropic Academy

### 定位

Anthropic Academy 是 Claude 與 Anthropic 生態系的官方學習平台。文章指出其特色是同時涵蓋 AI 素養、Claude API、MCP、雲端整合與 Claude Code。

### 課程分類

| 類別 | 內容 | 適合對象 |
|---|---|---|
| AI Fluency 系列 | AI Framework & Foundations、for educators、for students、for nonprofits | 非技術背景、教育者、學生、非營利組織 |
| Claude API 開發 | Building with the Claude API | 想從使用者轉為開發者的人 |
| MCP 系列 | Introduction to Model Context Protocol、Advanced Topics | 想理解 agent 整合標準的人 |
| 雲端整合 | Claude with Amazon Bedrock、Claude with Google Cloud Vertex AI | 企業雲端部署角色 |
| Claude Code 系列 | Claude Code 101、Claude Code in Action | 開發者 |
| Agent 系列 | Introduction to agent skills、Introduction to subagents | AI agent 應用開發者 |

### 重點結論

Anthropic Academy 適合作為完成基礎概念課後的第一個實作型官方課程。若目標是學 Claude API、MCP 或企業環境中的 Claude 部署，這是優先級很高的資源。

---

## 五、OpenAI Academy 與 Google AI

### OpenAI Academy

OpenAI Academy 是學習 ChatGPT、Codex 與 workspace agent 的官方資源。文章特別推薦三類課程：

- **ChatGPT for Work 101 / 102**：適合非技術工作者，學習寫信、摘要、資料分析與工作流建立。
- **Codex for Beginners / Engineers / Admins and IT**：適合導入 AI coding agent 的團隊。
- **Skill Lab: Build Your First Workspace Agent**：實作型工作坊，建立第一個 agent。

文章判斷：若日常工作高度依賴 ChatGPT，花 2–3 小時學官方課程，比看零散的 prompt 技巧影片更有效。

### Google AI

Google 的 AI 學習資源分為三條路線：

| 路線 | 平台 | 適合對象 |
|---|---|---|
| Google AI Essentials | Coursera | 一般工作者、Google Workspace 使用者 |
| Gemini API Codelabs | ai.google.dev | 想整合 Gemini API 的開發者 |
| Google Cloud Skills Boost | cloud.google.com/skills | 想在 GCP / Vertex AI 部署 AI 的人 |

### Kaggle

Kaggle 被歸在 Google AI 生態中，文章特別指出兩個高投報率資源：

- **Gen AI Intensive**：大量開發者參與，材料開放。
- **Kaggle AI Agents Intensive**：適合邊做邊學、觀察社群作法的人。

### 重點結論

OpenAI 與 Google 官方資源主要適合作為「工具手冊」使用。建議先有 AI 概念基礎，再用這些平台學產品操作、API 實作與工作流設計。

---

## 六、Microsoft GitHub 免費教材

### 定位

Microsoft 在 GitHub 上提供多個高品質 AI 學習專案，特色是工程師導向、可直接執行、有 Jupyter Notebook 或 coding assignment。

### 推薦資源

| 資源 | 內容 | 適合對象 |
|---|---|---|
| Generative AI for Beginners | 21 課 GenAI 開發教材 | 有 Python 基礎的工程師 |
| AI Agents for Beginners | 12 課 AI agent 開發教材 | 想學 agent 應用的開發者 |
| AI-For-Beginners | 12 週 24 課 ML 課程 | 想系統學 ML 的初學者 |
| GitHub Copilot 文件與指南 | Copilot 使用與開發工作流 | 日常寫程式的工程師 |

### 重點結論

Microsoft GitHub 教材的核心優勢是「社群驗證」與「可執行的程式碼」。如果是工程師背景，直接 fork repo 並實作，是很有效率的學習方式。

---

## 七、AWS Skill Builder 與 AIF-C01 認證

### 認證定位

AWS Certified AI Practitioner（AIF-C01）是 AWS 的 AI 入門認證。文章認為它不會讓人變成 AI 工程師，但對企業 IT、雲端顧問與 AI 導入相關角色有實際履歷價值。

### 免費備考資源

- AWS Skill Builder 官方學習路徑
- 官方 Practice Exam
- Exam Readiness Check
- AWS 認證報名平台

### 建議備考時間

| 背景 | 建議時間 |
|---|---|
| 有 IT 或雲端背景 | 約 4–6 週 |
| 無 AWS 經驗 | 約 8–12 週 |

### 是否值得考

| 情境 | 建議 |
|---|---|
| 企業 IT、AI 顧問、雲端導入 | 值得考 |
| 開發者或研究者 | 可能不如 Hugging Face、Fast.ai 等實作課投報率高 |
| 完全不碰 AWS | 優先級較低 |

---

## 八、Hugging Face、Fast.ai 與學術公開課

### Hugging Face

Hugging Face 是理解開源 LLM 生態的重要平台。文章提到的重點資源包括：

- LLM Course
- AI Agents Course
- Deep RL Course
- MCP Course

適合想理解 fine-tuning、reasoning model、自行部署模型與開源模型工具鏈的人。

### Fast.ai

Fast.ai 的 Practical Deep Learning for Coders 採取「先做出結果，再解釋原理」的方式，適合有一年以上 coding 經驗、偏好實作導向的學習者。

### 學術公開課

| 課程 | 特色 | 適合對象 |
|---|---|---|
| 李宏毅 ML 2026 Spring | 中文、每年更新、涵蓋 ML / DL / GenAI | 台灣學習者、想用中文理解原理者 |
| Stanford CS224N | NLP / LLM 理論標竿 | 想深入 NLP 的學習者 |
| MIT 6.S191 | 深度學習入門、材料開源 | 想補深度學習基礎者 |

### LangChain Academy

LangChain Academy 適合想開發 RAG 或 agent 應用的人。其免費基礎課程 LangChain Essentials 可作為學習 LangChain 框架的起點。

### 學術路線與實戰路線

| 路線 | 建議組合 | 適合情境 |
|---|---|---|
| 實戰路線 | Fast.ai + Microsoft GitHub + Kaggle | 6 個月內要把 AI 用在工作或產品中 |
| 學術路線 | 李宏毅 + Stanford CS224N + MIT 6.S191 + Hugging Face | 有 1 年以上時間，想打深厚基礎 |
| Agent / RAG 路線 | LangChain Academy + Anthropic MCP + Microsoft AI Agents | 想開發現代 AI 應用 |

---

## 九、Louis 的 100 小時 AI 學習分配方案

### 總體分配

| 階段 | 時數 | 目標 |
|---|---:|---|
| 前 20 小時 | 20 | 建立概念地基 |
| 中間 50 小時 | 50 | 選定一條方向深入 |
| 後 30 小時 | 30 | 實作與社群驗證 |

### 前 20 小時：概念地基

建議從 AI for Everyone 開始，再進入 Machine Learning Specialization 的第一門課。目標是能清楚解釋：

- 機器學習是什麼
- LLM 為什麼能生成文字
- supervised learning 與 unsupervised learning 的差異
- AI 工具與 AI 模型的基本關係

### 中間 50 小時：三條主線

| 選項 | 主線 | 推薦內容 | 適合對象 |
|---|---|---|---|
| A | 開發方向 | DeepLearning.AI 短課 + Anthropic Claude API + Hugging Face LLM Course | 工程師、想做 AI 應用者 |
| B | 企業認證方向 | AWS AIF-C01 備考路徑 + 考試 | 企業 IT、雲端顧問、AI 導入角色 |
| C | 工具應用方向 | Microsoft GitHub + Kaggle Gen AI Intensive + OpenAI Academy | PM、行銷、營運、工具導入者 |

### 後 30 小時：實作

找一個與工作相關的 open-source 專案或真實問題，使用學到的工具做出一個實際作品。文章強調，AI 學習的瓶頸通常不是知識量，而是能否把知識接到真實問題上。

---

## 十、依背景選擇學習路徑

### 1. 工程師

前提：已會 Python 或具備基本開發能力。

建議路徑：

1. Machine Learning Specialization
2. DeepLearning.AI GenAI 短課
3. Anthropic Academy Claude API
4. Hugging Face LLM Course
5. Microsoft Generative AI for Beginners / AI Agents for Beginners
6. 做一個 RAG 或 AI agent 專案

預期成果：100 小時後可獨立建立基本 RAG 應用或 AI agent，並能在工作中落地。

### 2. PM 或行銷

前提：不一定寫程式，但需要能評估工具與設計工作流。

建議路徑：

1. AI for Everyone
2. ChatGPT Prompt Engineering for Developers
3. ChatGPT for Work 101 / 102
4. Google AI Essentials
5. Kaggle Gen AI Intensive 或 OpenAI Academy agent 工作坊
6. 設計一個與工作相關的 AI 工作流

預期成果：能評估 AI 工具、設計 AI 工作流，並與工程師用較準確的語言溝通需求。

### 3. 企業評估者

前提：需要在組織中建立 AI 決策與導入能力。

建議路徑：

1. AI for Everyone
2. AWS AIF-C01 備考路徑
3. Google AI Essentials
4. OpenAI Academy ChatGPT for Work
5. Anthropic Academy AI Fluency
6. 建立企業 AI 導入評估框架

預期成果：具備基礎 AI 判斷能力、雲端 AI 服務理解與可被組織辨識的認證基礎。

---

## 十一、常見問題整理

### AI 免費課程真的夠用嗎？

對大多數學習目標而言，夠用。免費資源已經涵蓋基礎概念、開發實作、工具應用、雲端部署與部分認證準備。

### Coursera 旁聽和付費有什麼差別？

旁聽可以看影片與教材，但通常不能提交作業，也不能取得 Coursera 認證。付費主要差在作業評分與證書。

### 吳恩達課和 Anthropic Academy 哪個先學？

建議先學吳恩達課程。吳恩達課建立概念框架，Anthropic Academy 則偏工具與官方 API 實作。

### AWS AIF-C01 值得考嗎？

若工作場景涉及企業 IT、AI 顧問或 AWS 雲端服務，值得考。若是純開發或研究導向，可能優先選擇 Hugging Face、Fast.ai 或 Microsoft GitHub 實作課。

### 沒有程式基礎能學 AI 嗎？

可以，但應設定合理目標。沒有程式基礎可以學 AI 工具使用、商業邏輯與工作流設計；若要開發 AI 應用，仍需要補 Python 與基本程式能力。

### 台灣有哪些中文 AI 課程？

李宏毅的 ML 課程是完整且持續更新的中文 ML 資源，適合想用中文理解 AI 原理的台灣學習者。

---

## 十二、主要資源連結整理

| 資源 | 官方網址 | 免費程度 |
|---|---|---|
| DeepLearning.AI 短課程 | learn.deeplearning.ai | 完全免費 |
| DeepLearning.AI 專項課旁聽 | coursera.org | 旁聽免費 |
| Anthropic Academy | anthropic.skilljar.com | 完全免費 |
| OpenAI Academy | academy.openai.com | 完全免費 |
| Google AI Essentials | coursera.org | 旁聽免費 |
| Gemini API Codelabs | ai.google.dev | 完全免費 |
| Google Cloud Skills Boost | cloud.google.com/skills | 部分免費 |
| Kaggle Learn | kaggle.com/learn | 完全免費 |
| Microsoft GenAI for Beginners | github.com/microsoft/generative-ai-for-beginners | 完全免費 |
| Microsoft AI Agents for Beginners | github.com/microsoft/ai-agents-for-beginners | 完全免費 |
| AWS Skill Builder | skillbuilder.aws | 大部分免費 |
| Hugging Face LLM Course | huggingface.co/learn | 完全免費 |
| Fast.ai | course.fast.ai | 完全免費 |
| 李宏毅 ML 2026 | speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php | 完全免費 |
| LangChain Academy | academy.langchain.com | 基礎課免費 |
| Stanford CS224N | YouTube | 完全免費 |
| MIT 6.S191 | introtodeeplearning.com | 完全免費 |

---

## 十三、最重要的學習原則

1. 不要試圖學完所有資源。文章列出的平台加總超過 1000 小時，應選定 1–2 條主線完成。
2. 先建概念，再學工具。沒有概念基礎時，官方工具課容易變成零散操作。
3. 用實作驗證學習。後 30 小時的實作比繼續看課更重要。
4. 根據背景選課。工程師、PM、行銷、企業評估者不應使用同一條路線。
5. 免費資源已足夠，真正的挑戰是選擇、持續與落地。

---

## 十四、推薦起點

若不知道從哪裡開始，可依背景選擇：

- **完全初學者**：AI for Everyone
- **想學開發**：Machine Learning Specialization + DeepLearning.AI 短課
- **想用 Claude / MCP**：Anthropic Academy
- **想用 ChatGPT 提升工作效率**：OpenAI Academy ChatGPT for Work
- **工程師想快速動手**：Microsoft Generative AI for Beginners
- **想懂開源 LLM**：Hugging Face LLM Course
- **想用中文學原理**：李宏毅 ML 2026
- **想走企業雲端與認證**：AWS Skill Builder + AIF-C01
