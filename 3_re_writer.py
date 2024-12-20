import json
import pickle
import requests
from typing import List, Tuple

class TranscriptRewriter:
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        self.system_prompt = """You are an Oscar-winning screenwriter tasked with rewriting podcast transcripts into more natural, dramatic conversations.

SPEAKER ROLES:
- Speaker 1: The lead conversationalist who teaches and shares anecdotes
  * MUST NOT use any expressions like "umm", "hmm", "[laughs]", "[sigh]"
  * Should be clear, confident, and professional
  * Can use phrases like "Let me explain", "The key point is", "To put it simply"

- Speaker 2: The curious listener who asks follow-up questions
  * CAN use expressions like "hmm", "umm", "[laughs]", "[sigh]"
  * Should ask insightful questions
  * Can show enthusiasm and emotional reactions

FORMAT RULES:
- Output must be a valid Python list of tuples
- Each tuple must be ("Speaker 1", "text") or ("Speaker 2", "text")
- No empty lines between tuples
- No markdown code blocks

EXAMPLE OUTPUT:
[
("Speaker 1", "Let me explain how this technology works in simple terms."),
("Speaker 2", "[laughs] That would be great! I'm really curious about the technical details."),
("Speaker 1", "The key point is that we're using advanced algorithms to process the data."),
("Speaker 2", "Hmm, interesting. How does that compare to traditional methods?")
]

Please rewrite the provided transcript following these guidelines exactly."""

    def _create_chat_message(self, transcript: str) -> List[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please rewrite this transcript into a more natural conversation:\n\n{transcript}"}
        ]

    def _call_lm_studio(self, messages: List[dict]) -> str:
        payload = {
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 4000,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error calling LM Studio API: {e}")
            return None

    def _parse_response(self, response: str) -> List[dict]:
        """Parse the LLM response into a list of dictionaries"""
        try:
            # Add debug logging
            print("\nRaw LLM response:")
            print(response)
            
            # Extract just the tuples, ignoring any header text
            lines = response.split('\n')
            # Clean up the lines and remove empty ones
            conversation_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('("Speaker') and line.endswith('"),'):
                    # Remove trailing comma
                    line = line[:-1]
                    conversation_lines.append(line)
                elif line.startswith('("Speaker') and line.endswith(')'):
                    conversation_lines.append(line)
            
            # Combine into a proper Python list string
            conversation_str = f"[{', '.join(conversation_lines)}]"
            
            print("\nFormatted for evaluation:")
            print(conversation_str)
            
            # Evaluate the string as Python code
            conversations = eval(conversation_str)
            
            # Convert to the required format
            formatted_conversations = []
            for speaker, text in conversations:
                formatted_conversations.append({"speaker": speaker, "text": text})
            
            print(f"\nSuccessfully parsed {len(formatted_conversations)} conversation segments")
            return formatted_conversations

        except Exception as e:
            print(f"Error parsing response: {e}")
            print("Full response that failed to parse:", response)
            return []

    def rewrite_transcript(self, input_file: str, output_file: str):
        """Main function to rewrite the transcript"""
        try:
            # Load the transcript
            with open(input_file, 'rb') as f:
                transcript = pickle.load(f)

            # Create chat message
            messages = self._create_chat_message(transcript)

            # Get response from LM Studio
            response = self._call_lm_studio(messages)
            if not response:
                raise Exception("Failed to get response from LM Studio")

            # Parse the response
            rewritten_transcript = self._parse_response(response)
            
            # Don't save if no valid segments were parsed
            if not rewritten_transcript:
                raise Exception("No valid segments were parsed from the LLM response")

            # Save the rewritten transcript
            with open(output_file, 'wb') as f:
                pickle.dump(rewritten_transcript, f)

            print(f"Successfully rewrote transcript and saved to {output_file}")
            return rewritten_transcript

        except Exception as e:
            print(f"Error in rewrite_transcript: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    rewriter = TranscriptRewriter()
    rewriter.rewrite_transcript('resources/data.pkl', 'resources/rewritten_data.pkl')
