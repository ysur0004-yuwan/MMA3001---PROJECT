# PROJECT NAME

An Automated Detection of Pork Rasher Packaging Errors - MMA3001 PROJECT 

A computer vision system that classifies images of pork rasher trays as a pass or a fail based on the presence of a packaging or product defect, with the extension to classify the specific defect type, loose meat, twisted meat, unsealed packaging, wrinkles, and supporting a human in the loop for the quality inspection decision. 


# FEATURES

Binary image classification, a pass or fail based on the presence of a packaging defect. 

Multi class classification to identify the specific defect type where a defect is present 

Image level labels derived from the data sets original bounding box annotations 

Baseline vs an improved model comparison 

Validation against a predefined train, validation, split


# TECH STACK 

Framework being used is Tensor Flow  using transfer learning 
Environment is on google colab 
Dataset tooling from roboflow, converted from bounding box to image level labels for classification 


# LICENSES 

Note on dataset licensing: the training data is sourced from the "Pork Rasher Error Packaging" Dataset on Roboflow Universe. See citation below. Released under a CC BY 4.0 License. The data set is not redistributed in this repository. 

# CITATION 

Dataset: Pork Rasher Error (Packaging), by Hello, Roboflow Universe, 2025. 
Available at: https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging Licence: CC BY 4.0


# AI ACKNOWLEDGEMENT 

This project's development incorporated assistance from various Artificial Intelligence (AI) tools to enhance efficiency and explore potential solutions. This statement outlines the specific AI tools used, their application, and the human oversight exercised to ensure the quality and integrity of the work, in accordance with Monash University guidelines.

### AI Tools Used

*   **[Name of AI Tool 1] (e.g., via Google Colab)**: Utilized for [brief description of primary use, e.g., code generation, debugging, brainstorming]. (Accessed: [Start Date] - [End Date], Model: [Specific AI Model or Version, e.g., Gemini 2.5 LLM preview]).
*   **[Name of AI Tool 2]**: Employed for [brief description of primary use, e.g., generating initial drafts of documentation, rephrasing text, suggesting naming conventions]. (Accessed: [Start Date] - [End Date], Model: [Specific AI Model or Version]).
*   **[Add more as needed]**

### How AI Was Used

AI tools were employed as intellectual assistants and accelerators for specific tasks, always under direct human supervision:

*   **Code Generation**: For routine programming tasks, boilerplate code, and exploring different algorithmic implementations. AI-generated code snippets were always reviewed, tested, and adapted to fit the project's specific requirements.
*   **Documentation & Explanation**: To assist in drafting explanatory text, summarizing complex concepts, and ensuring clear communication in project documentation, including this README file.
*   **Problem-Solving & Brainstorming**: AI provided alternative perspectives or solutions to design challenges, serving as a sounding board for problem identification and resolution.
*   **[Add more specific examples as needed, e.g., Data analysis, visualization conceptualization, error identification].**

### Human Oversight and Responsibility

Crucially, all AI-generated content was subjected to rigorous human review, validation, and adaptation. The project team maintained complete oversight and responsibility for the final output. This involved:

*   **Verification**: Independently verifying the accuracy, correctness, and relevance of all AI-generated code, text, and suggestions.
*   **Adaptation**: Modifying and integrating AI outputs to align with project standards, ethical considerations, and desired functionality.
*   **Critical Evaluation**: Actively scrutinizing AI responses for potential biases, inaccuracies, or logical flaws, especially in areas requiring nuanced understanding or creative solutions.

We acknowledge that while AI significantly aided in aspects of this project, the ultimate responsibility for the conclusions, design decisions, code functionality, and overall quality rests with the human developers. This approach ensures that the project adheres to the highest standards of professional engineering practice and academic integrity.

***
**Author's Note:** The human author maintains full accountability for the accuracy and quality of the outputs presented in this project, and for ensuring that the final product is fit for purpose and meets the professional standards expected of an engineer.
