
# ContentQueryBot: Chatbot with Document and Web URL Support

## Overview

This project is a Streamlit-based application that allows users to interact with various documents (PDF, DOCX) and web URLs using a chatbot interface. The chatbot uses the `facebook/bart-large-cnn` model for summarization to provide concise and relevant answers to user queries based on the content of the uploaded documents or entered web URLs.

## Features

- Upload PDF and DOCX files.
- Enter a web URL to fetch text content.
- Process the uploaded files and web content.
- Ask questions about the content and receive summarized answers.
- Interactive and user-friendly interface using Streamlit.

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` package manager

### Required Libraries

To install the required libraries, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Sripriyan/ContentQueryBot.git
    cd ContentQueryBot
    ```

2. **Run the Application**

    ```bash
    streamlit run chatbot.py
    ```

3. **Upload Files or Enter URL**

    - Use the sidebar to upload PDF or DOCX files.
    - Or enter a web URL to fetch text content.

4. **Ask Questions**

    - After processing the documents or URL, you can ask questions about the content.
    - The chatbot will provide summarized answers based on the processed content.

## Custom CSS

The `styles.css` file is used to apply custom styles to the Streamlit application. You can modify it to change the look and feel of the app as per your preference.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
