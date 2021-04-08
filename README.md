# gencon-nlp
A school project for analyzing general conference talks (from the Church of Jesus Christ of Latter Day Saints) using Natural Language Processing.

The project structure is as follows:
- analysis: data visualization and analysis using Gibbs LDA
- data_processing: Processing of HTML files obtained from web scraping
- Feature_Engineering: Further processing of data, such as determining which speakers have ever served as apostles
- figures: Image files of figures produced during analysis of the data
- HMM: Application of Hidden Markov Models to speaker and topic recognition
- Kaplan_Meier: Application of Kaplan Meier estimation to predicting how far into a talk a speaker is likely to use a word or phrase
- lsi: Application of Latent Semantic Indexing to find most and least similar talks
- scraping: Code that we used to scrape talks