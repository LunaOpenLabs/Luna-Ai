# Luna AI

**Luna AI** is an open-source AI model developed by Luna OpenLabs, designed for text classification tasks. Utilizing the powerful BERT architecture, Luna AI efficiently and accurately classifies text into predefined categories. In addition, Luna AI now features an interactive chat interface, allowing you to easily test the model with real-time classification and conversation responses.

Luna AI is available for integration, development, and deployment, offering flexibility for users and developers alike.

## Table of Contents

- [Features](#features)
- [Logics and Capabilities](#logics-and-capabilities)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Chat Interface Usage](#chat-interface-usage)
  - [Saving and Loading the Model](#saving-and-loading-the-model)
  - [Testing the Model](#testing-the-model)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Text Classification**: Classify text into various predefined categories.
- **Chat Interface**: Test the model using an interactive chat interface to classify your data and generate conversational responses.
- **Built on BERT**: Uses BERT's architecture for powerful natural language understanding and text classification.
- **Real-time Interaction**: The chat interface allows real-time interaction with the model for conversational queries.
- **Open Source**: The entire project is open source and available for anyone to use, modify, and contribute to.

## Logics and Capabilities

Luna AI is equipped with a variety of logic modules and capabilities to handle different natural language processing tasks. Here’s a list of what Luna AI can do:

1. **Text Classification**:
   - Classifies text into predefined categories such as sentiment analysis, spam detection, product reviews, etc.
   
2. **Sentiment Analysis**:
   - Detects whether a given text expresses positive, negative, or neutral sentiment.
   
3. **Text Summarization**:
   - Summarizes long texts or documents into shorter, concise summaries that retain the main points.
   
4. **Text Generation**:
   - Generates coherent and contextually relevant text based on a given prompt.
   
5. **Language Translation**:
   - Translates text from one language to another (e.g., English to Spanish or German to English).

6. **Named Entity Recognition (NER)**:
   - Identifies and categorizes entities like names, dates, locations, and more in the text.

7. **Question Answering**:
   - Provides answers to questions based on the context provided. It extracts relevant information from a body of text to answer user queries.

8. **Text Correction**:
   - Detects spelling and grammatical errors in text and suggests corrections.

9. **Paraphrase Generation**:
   - Rewrites input text in different words while retaining the original meaning.

10. **Keyword Extraction**:
    - Extracts the most relevant and important keywords from a given piece of text.

11. **Emotion Detection**:
    - Identifies the underlying emotions in text, such as happiness, anger, sadness, etc.

12. **Intent Recognition**:
    - Determines the intent behind a given text, useful for conversational AI and chatbots.

13. **Topic Modeling**:
    - Identifies the main topics in a given document or set of documents.

14. **Text-to-Speech (TTS)**:
    - Converts text input into speech, allowing for audio responses from the model.

15. **Speech-to-Text (STT)**:
    - Converts spoken language into written text, enabling voice-based interaction with the model.

16. **Text-Based Recommendations**:
    - Provides recommendations based on text input, such as suggesting movies, books, or products based on user preferences.

17. **Semantic Textual Similarity**:
    - Measures the similarity between two text inputs, useful for matching similar sentences or paraphrases.

18. **Spell Check and Auto-correction**:
    - Automatically detects and corrects spelling mistakes in user input.

19. **Fact-checking**:
    - Verifies the accuracy of a statement by comparing it against reliable sources or knowledge bases.

20. **Custom Classification Logic**:
    - Users can fine-tune Luna AI for domain-specific classification tasks based on their custom datasets.

21. **Text Embedding**:
    - Converts text into numerical vectors that capture semantic meaning, useful for machine learning tasks.

22. **Conversational Agent**:
    - Provides coherent and contextually relevant replies in a conversational manner.

23. **Summarizing Chat Logs**:
    - Summarizes ongoing conversations, useful for chatbots or customer support systems.

24. **Handling Multiple Languages**:
    - Luna AI supports multilingual inputs and can perform classification tasks across different languages.

25. **Contextual Understanding**:
    - Maintains context across multiple user inputs in a conversation for more accurate responses.

26. **Textual Data Filtering**:
    - Identifies and filters out irrelevant or inappropriate content from text inputs.

27. **Data Cleaning**:
    - Cleanses raw textual data by removing noise, unwanted characters, and irrelevant details.

28. **Plagiarism Detection**:
    - Checks whether a piece of text has been plagiarized or is similar to existing content on the web.

29. **Text-based Data Mining**:
    - Extracts valuable insights and patterns from large collections of text data.

30. **Speech Recognition**:
    - Converts spoken language into text, enabling voice commands and speech-based inputs.

31. **Content Moderation**:
    - Detects harmful or offensive content, useful for social media, forums, or online communities.

32. **Tagging and Labeling**:
    - Automatically tags or labels text based on predefined categories (e.g., genre, topic).

33. **Customized Response Generation**:
    - Generates personalized replies based on the user’s input, useful for AI-powered chatbots.

34. **Event Detection**:
    - Detects events or significant occurrences mentioned in text, such as incidents, actions, or milestones.

35. **Document Classification**:
    - Categorizes documents based on predefined tags, such as news articles, research papers, etc.

36. **Text Filtering for Compliance**:
    - Automatically filters content for regulatory compliance, such as in financial or healthcare texts.

37. **Optical Character Recognition (OCR)**:
    - Extracts text from images, making scanned documents or images of text usable for NLP tasks.

38. **Document Clustering**:
    - Groups similar documents together based on their content for easier classification or search.

39. **Pattern Recognition**:
    - Identifies recurring patterns or trends within textual data for analysis.

40. **Customizable Task Handling**:
    - Luna AI can be trained on custom tasks such as specific content generation, classification, or summarization.

41. **Context-Aware Text Generation**:
    - Generates context-aware responses, taking into account the broader context of previous inputs.

42. **Turing Test Interaction**:
    - Mimics human-like conversation to test the AI's ability to engage in natural interactions.

43. **Collaborative Filtering**:
    - Provides recommendations based on user behavior and preferences, useful for e-commerce or content platforms.

44. **Error Detection in Code**:
    - Detects errors or issues in programming code, useful for developers.

45. **Automated Reporting**:
    - Automatically generates summaries and reports based on input data or ongoing interactions.

46. **Adaptive Learning**:
    - The AI learns and adapts to new data inputs over time, improving classification accuracy.

47. **Interactive Storytelling**:
    - Generates interactive narratives or stories based on user input and previous interactions.

48. **Custom Query Resolution**:
    - Provides accurate answers to custom queries that are outside the predefined scope.

49. **Real-Time Language Detection**:
    - Detects the language of user inputs in real time for multilingual support.

50. **Social Media Sentiment Analysis**:
    - Analyzes sentiments of social media posts, providing insights into public opinions and trends.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- [Flask](https://flask.palletsprojects.com/) (for the web application)

### Clone the Repository

Clone the repository using the following command:

```bash
git clone https://github.com/LunaOpenLabs/Luna-Ai.git
```

### Install Requirements

To install the required Python libraries, use:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies, including Flask and Hugging Face Transformers.

---

## Dataset

Luna AI requires a dataset in CSV format for training the model. The dataset must contain at least two columns: `text` and `label`.

### Example Dataset Structure

Here’s an example of the dataset format:

```csv
text,label
"I love this product!",1
"This is the worst experience.",0
```

You can find an example dataset in the `data/` directory of this repository.

---

## Usage

### Training the Model

To train the model, execute the following command:

```bash
python training/train.py
```

This command will load the dataset from `data/data.csv` and initiate the training process.

### Chat Interface Usage

To run the interactive chat interface, use the following:

```bash
python app.py
```

This will start the Flask server, and you can interact with Luna AI via the browser.

### Saving and Loading the Model

After training, save the trained model using:

```bash
python save_model.py
```

This will save the model and its tokenizer in the `luna_ai_model` directory.


## Contributing

We welcome contributions! To contribute to Luna AI, please fork the repository and submit a pull request with your improvements or fixes.

---

## License

Luna AI is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For questions, support, or feedback, you can reach us at:

- **Email**: lunaopenlabs@outlook.com
- **GitHub**: [Luna OpenLabs GitHub](https://github.com/LunaOpenLabs)
