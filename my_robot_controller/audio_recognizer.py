import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import sounddevice as sd
import threading
import re
from whisper_mic.whisper_mic import WhisperMic

class AudioRecognizerNode(Node):
    def __init__(self):
        super().__init__('audio_recognizer_node')
        self.publisher_ = self.create_publisher(String, 'recognized_instruction', 10)
        self.feedback_publisher_ = self.create_publisher(String, 'system_feedback', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Initialize WhisperMic for transcription using the "medium.en" model
        self.whisper_mic = WhisperMic(model="small.en", device="cpu")

        # Get the full path for the model
        current_path = os.getcwd()
        model_name = 'bert_model'
        model_path = os.path.join(current_path, model_name)
        
        # Initialize BERT model and tokenizer for instruction classification
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval() 
        
        # Set the device to CPU
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Confidence threshold for instruction classification
        self.CONFIDENCE_THRESHOLD = 0.65
        
        # Flag to control when to listen
        self.listen_for_instruction = True

        # Adding a lock to prevent concurrent execution of record_and_process
        self.recording_lock = threading.Lock()

        # Inform the system is ready to listen
        self.publish_feedback("System is ready to listen.")

    def publish_feedback(self, message):
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_publisher_.publish(feedback_msg)
        self.get_logger().info(message)

    def timer_callback(self):
        if self.listen_for_instruction:
            # Acquire the lock before starting a new thread
            if self.recording_lock.acquire(blocking=False):
                try:
                    threading.Thread(target=self.record_and_process).start()
                except Exception as e:
                    self.get_logger().error(f'Failed to start recording thread: {e}')
                    self.recording_lock.release()

    def record_and_process(self):
        self.publish_feedback("Recording and processing speech.")
        try:
            # Use WhisperMic to capture and transcribe audio
            transcription = self.whisper_mic.listen()
            self.get_logger().info(f"Transcribed text: {transcription}")
            
            # Extract value if present in the text using regex
            value_match = re.search(r'\b\d+(\.\d+)?', transcription)
            value = float(value_match.group(0)) if value_match else None
            
            # Classify the instruction using BERT
            instruction, confidence = self.classify_instruction(transcription)
            
            # Log the confidence level regardless of the threshold
            self.get_logger().info(f"Confidence level: {confidence:.2f}")

            # Check confidence level and publish if above threshold
            if confidence >= self.CONFIDENCE_THRESHOLD:
                self.publish_instruction(transcription, instruction, confidence, value)
            else:
                self.publish_feedback(f"Warning: Low confidence prediction ({confidence:.2f}). Command might be ambiguous.")
                         
        except Exception as e:
            self.get_logger().error(f'An error occurred during recording and processing: {e}')
        finally:
            # Release the lock
            self.recording_lock.release()
            self.reset_listener()

    def classify_instruction(self, text):
        # Tokenize and process the text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1)
            confidence = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_label].item()

        # Map the predicted label to the instruction
        label_to_instruction = {
            0: "take off",
            1: "land",
            2: "go up",
            3: "go down",
            4: "go forward",
            5: "go backward",
            6: "move right",  
            7: "move left",
            8: "turn right",
            9: "turn left",
            10: "stop"
        }
        instruction = label_to_instruction.get(predicted_label.item(), "unknown")

        return instruction, confidence

    def publish_instruction(self, transcribed_text, instruction, confidence, value):
        instruction_text = f"{instruction} {value}" if value is not None else instruction
        msg = String()
        msg.data = instruction_text
        self.publisher_.publish(msg)
        self.publish_feedback(f"Transcribed text: '{transcribed_text}'\nRecognized instruction: {instruction_text} with confidence {confidence:.2f}")

    def reset_listener(self):
        # Reset the listener to start listening for instructions again
        self.listen_for_instruction = True
        self.publish_feedback('Listener has been reset and is ready to listen for instructions.')

def main(args=None):
    rclpy.init(args=args)
    audio_recognizer_node = AudioRecognizerNode()
    rclpy.spin(audio_recognizer_node)
    audio_recognizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
