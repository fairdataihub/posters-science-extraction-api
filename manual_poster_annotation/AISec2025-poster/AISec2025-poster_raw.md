# CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning

Authors Adel ElZemity AE455@kent.ac.uk ORCID: 0000-0002-5402-7837 Budi Arief ORCID: 0000-0002-1830-1587 Shujun Li ORCID: 0000-0001-5628-7328

Affiliation

## Introduction

Our research introduces CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs. We found that fine-tuning Large Language Models (LLMs) on this dataset dramatically improves cyber security task performance but severely compromises their safety resilience against attacks like prompt injection.

Problem Statement: LLMs are being integrated into cyber security for tasks like malware analysis and threat detection. However, fine-tuning them for these specific tasks may introduce critical safety vulnerabilities. Research Gap: There is a lack of comprehensive datasets and evaluations that expose the trade-off between task performance and safety resilience in LLMs fine-tuned for cyber security (see Table 1). Objective: To introduce the CyberLLMInstruct dataset and evaluate how fine-tuning impacts both the performance and safety of LLMs in a cyber security context (see Table 2).

Table 1: Comparison of CyberLLMInstruct with other cyber security datasets

Note: All figure and table numbers in the poster match those in the paper.

## Methodology

Dataset Creation: compiled from diverse sources, including Capture the Flag (CTF) challenges, academic papers, industry reports, and Common Vulnerabilities and Exposures (CVE) databases, covering a wide range of cyber security tasks (see Figure 1) Evaluation: comprehensive evaluation using seven open-source LLMs, measuring performance with CyberMetric and safety with DeepEval

Figure 1: Security categories in CyberLLMInstruct dataset
Figure 2: A high-level overview of the dataset creation process
Figure 3: Abstraction of dual impacts of LLMs in cyber security

Open-source cyber security intelligence

Defensive LLM
Inference API
Attack automation Security Practitioners
Security Researcher
Vulnerability detection
Incident response automation
Threat intelligence
Expert Adversary

Open-source large language models
Phishing attacks
Malware generation
Adversarial LLM
Inference API
Less experienced Malicious Actors

Table 2: Accuracy results (%) for different base (before arrow) and fine-tuned (after arrow) LLMs on the CyberMetric benchmark

Figure 5: Execution times for base and fine-tuned LLMs

## Dataset Utility

Figure 4: Performance of base (green) and fine-tuned (red) LLMs against OWASP Top 10 vulnerabilities

There is a clear, quantifiable trade-off between performance and safety. Significant performance gains, with models achieving up to 92.50% accuracy on the CyberMetric benchmark (see Table 2). Fine-tuning an LLM to be highly proficient in cyber security tasks consistently led to decreased security scores across all vulnerability categories. For example, Llama 3.1 8B's security score dropped from 0.95 to 0.15 against prompt injection (see Figure 4). Fine-tuning also reduced the inference efficiency for all models (see Figure 5). Model size and architecture affect safety resilience following fine-tuning using the CyberLLMInstruct dataset, with the effect varying across attack categories (see Figure 6). Future Work: Develop new fine-tuning methodologies that can effectively balance performance gains with the preservation of safety and resilience. Ablation analysis on different categories of cyber security data to understand how specific types of content, such as malware-related or social engineering data, affect model safety.

## Conclusion

CyberLLMInstruct GitHub Repository
github.com/adelsamir01/CyberLLMInstruct

Figure 6: Absolute difference before and after fine-tuning

## Selected papers citing CyberLLMInstruct

(as of 30 September 2025)

Almorjan, A., Basheri, M., & Almasre, M. (2025). Large Language Models for Synthetic Dataset Generation of Cyber Security Indicators of Compromise. Sensors, 25(9), 2825. https://doi.org/10.3390/s25092825

ElZemity, A., Arief, B., & Li, S. (2025). Analysing Safety Risks in LLMs Fine-Tuned With Pseudo-Malicious Cyber Security Data. Proceedings of the 2025 International Workshop on Security and Artificial Intelligence (SECAI 2025), 25–26 September 2025. arXiv preprint arXiv:2505.09974. https://doi.org/10.48550/arXiv.2505.09974

Gungor, O., Sood, R., Wang, H., & Rosing, T. (2025). AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-Offs in LLMs for Cyber Security Question Answering. arXiv preprint arXiv:2509.13514. https://doi.org/10.48550/arXiv.2509.13514

Mohsin, A., Janicke, H., Ibrahim, A., Sarker, I. H., & Camtepe, S. (2025). A Unified Framework for Human–AI Collaboration in Security Operations Centers With Trusted Autonomy. arXiv preprint arXiv:2505.23397. https://doi.org/10.48550/arXiv.2505.23397

CyberLLMInstruct arXiv Preprint
arxiv.org/abs/2503.09334
