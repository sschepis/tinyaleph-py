"""
Tests for the input_templates module.

Run with: python -m pytest apps/reso_llm/test_input_templates.py -v
"""

import pytest
from apps.reso_llm.input_templates import (
    Role, Message, TemplateOutput, TokenizedOutput, SpecialTokens,
    InputTemplate, ChatTemplate, InstructionTemplate, QATemplate,
    CompletionTemplate, DialogueTemplate, CodeTemplate, PreferenceTemplate,
    TemplateRegistry, get_template, TemplatedDataset, apply_template_to_batch,
)


# ==============================================================================
# Mock Tokenizer for Testing
# ==============================================================================

class MockTokenizer:
    """Simple mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab = {}
        self.next_id = 0
        self.pad_token_id = 0
    
    def encode(self, text: str) -> list:
        """Simple character-level encoding for testing."""
        return [ord(c) for c in text]
    
    def decode(self, ids: list) -> str:
        """Decode token ids back to text."""
        return "".join(chr(i) for i in ids)


# ==============================================================================
# Test Data Structures
# ==============================================================================

class TestMessage:
    def test_message_creation(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.name is None
    
    def test_message_to_dict(self):
        msg = Message(role=Role.ASSISTANT, content="Hi there", name="bot")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
        assert d["name"] == "bot"
    
    def test_message_from_dict(self):
        data = {"role": "user", "content": "Test message"}
        msg = Message.from_dict(data)
        assert msg.role == Role.USER
        assert msg.content == "Test message"


class TestSpecialTokens:
    def test_default_tokens(self):
        tokens = SpecialTokens()
        assert tokens.bos == "<|bos|>"
        assert tokens.eos == "<|eos|>"
    
    def test_all_tokens(self):
        tokens = SpecialTokens()
        all_toks = tokens.all_tokens()
        assert len(all_toks) == 14
        assert "<|bos|>" in all_toks
    
    def test_get_role_tokens(self):
        tokens = SpecialTokens()
        start, end = tokens.get_role_tokens(Role.USER)
        assert start == "<|user|>"
        assert end == "<|/user|>"


# ==============================================================================
# Test Chat Template
# ==============================================================================

class TestChatTemplate:
    def test_validation_valid(self):
        template = ChatTemplate()
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        is_valid, error = template.validate(data)
        assert is_valid is True
        assert error is None
    
    def test_validation_missing_messages(self):
        template = ChatTemplate()
        is_valid, error = template.validate({})
        assert is_valid is False
        assert "messages" in error
    
    def test_validation_empty_messages(self):
        template = ChatTemplate()
        is_valid, error = template.validate({"messages": []})
        assert is_valid is False
    
    def test_format_simple(self):
        template = ChatTemplate()
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        output = template.format(data)
        assert isinstance(output, TemplateOutput)
        assert "<|user|>" in output.text
        assert "<|assistant|>" in output.text
        assert "Hello" in output.text
        assert "Hi there!" in output.text
    
    def test_format_with_system(self):
        template = ChatTemplate()
        data = {
            "system_prompt": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        output = template.format(data)
        assert "<|system|>" in output.text
        assert "You are helpful." in output.text
    
    def test_segments_loss_mask(self):
        template = ChatTemplate()
        data = {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer!"},
            ]
        }
        output = template.format(data)
        assert len(output.segments) == 2
        assert output.segments[0]["compute_loss"] is False  # User
        assert output.segments[1]["compute_loss"] is True   # Assistant
    
    def test_train_on_input(self):
        template = ChatTemplate(train_on_input=True)
        data = {"messages": [{"role": "user", "content": "Test"}]}
        output = template.format(data)
        assert output.segments[0]["compute_loss"] is True
    
    def test_callable_interface(self):
        template = ChatTemplate()
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        output = template(data)
        assert isinstance(output, TemplateOutput)
    
    def test_with_tokenizer(self):
        template = ChatTemplate()
        tokenizer = MockTokenizer()
        data = {"messages": [{"role": "user", "content": "Hi"}]}
        output = template(data, tokenizer)
        assert isinstance(output, TokenizedOutput)
        assert len(output.input_ids) > 0
        assert len(output.attention_mask) == len(output.input_ids)


# ==============================================================================
# Test Instruction Template
# ==============================================================================

class TestInstructionTemplate:
    def test_validation(self):
        template = InstructionTemplate()
        valid_data = {"instruction": "Do this", "output": "Done"}
        assert template.validate(valid_data)[0] is True
        
        invalid_data = {"instruction": "Do this"}
        assert template.validate(invalid_data)[0] is False
    
    def test_format_basic(self):
        template = InstructionTemplate()
        data = {"instruction": "Translate to French", "output": "Bonjour"}
        output = template.format(data)
        assert "Translate to French" in output.text
        assert "Bonjour" in output.text
    
    def test_format_with_input(self):
        template = InstructionTemplate()
        data = {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
        output = template.format(data)
        assert "Input: Hello" in output.text
        assert output.metadata["has_input"] is True
    
    def test_no_system(self):
        template = InstructionTemplate(include_system=False)
        data = {"instruction": "Test", "output": "Result"}
        output = template.format(data)
        assert "<|system|>" not in output.text


# ==============================================================================
# Test QA Template
# ==============================================================================

class TestQATemplate:
    def test_validation(self):
        template = QATemplate()
        valid = {"question": "What?", "answer": "This."}
        assert template.validate(valid)[0] is True
        
        invalid = {"question": "What?"}
        assert template.validate(invalid)[0] is False
    
    def test_format_basic(self):
        template = QATemplate()
        data = {"question": "What is 2+2?", "answer": "4"}
        output = template.format(data)
        assert "Question: What is 2+2?" in output.text
        assert "Answer: 4" in output.text
    
    def test_format_with_context(self):
        template = QATemplate()
        data = {"context": "Math facts.", "question": "What is 2+2?", "answer": "4"}
        output = template.format(data)
        assert "Context: Math facts." in output.text
        assert output.metadata["has_context"] is True
    
    def test_custom_prefixes(self):
        template = QATemplate(question_prefix="Q: ", answer_prefix="A: ")
        data = {"question": "Test?", "answer": "Yes"}
        output = template.format(data)
        assert "Q: Test?" in output.text
        assert "A: Yes" in output.text


# ==============================================================================
# Test Completion Template
# ==============================================================================

class TestCompletionTemplate:
    def test_validation_string(self):
        template = CompletionTemplate()
        assert template.validate("Some text")[0] is True
    
    def test_validation_dict(self):
        template = CompletionTemplate()
        assert template.validate({"prompt": "Start", "completion": "end"})[0] is True
        assert template.validate({"text": "Full text"})[0] is True
        assert template.validate({})[0] is False
    
    def test_format_string(self):
        template = CompletionTemplate()
        output = template.format("Complete this text")
        assert output.text == "Complete this text"
        assert output.metadata["mode"] == "pure"
        assert output.segments[0]["compute_loss"] is True
    
    def test_format_prompt_completion(self):
        template = CompletionTemplate()
        data = {"prompt": "Once upon a ", "completion": "time"}
        output = template.format(data)
        assert output.text == "Once upon a time"
        assert output.input_text == "Once upon a "
        assert output.target_text == "time"
    
    def test_prompt_suffix(self):
        template = CompletionTemplate(prompt_suffix="\n###\n")
        data = {"prompt": "Start", "completion": "End"}
        output = template.format(data)
        assert "Start\n###\nEnd" in output.text


# ==============================================================================
# Test Dialogue Template
# ==============================================================================

class TestDialogueTemplate:
    def test_validation(self):
        template = DialogueTemplate()
        assert template.validate("### Human: Hi")[0] is True
        assert template.validate({"text": "### Human: Hi"})[0] is True
        assert template.validate({"turns": []})[0] is True
    
    def test_parse_text(self):
        template = DialogueTemplate()
        text = "### Human: Hello\n### Assistant: Hi there"
        turns = template._parse_text(text)
        assert len(turns) == 2
        assert turns[0]["speaker"] == "Human"
        assert turns[1]["speaker"] == "Assistant"
    
    def test_format_from_text(self):
        template = DialogueTemplate()
        text = "### Human: Question?\n### Assistant: Answer!"
        output = template.format(text)
        assert "### Human: Question?" in output.text
        assert "### Assistant: Answer!" in output.text
    
    def test_format_from_turns(self):
        template = DialogueTemplate()
        data = {"turns": [
            {"speaker": "Human", "text": "Hello"},
            {"speaker": "Assistant", "text": "Hi"},
        ]}
        output = template.format(data)
        assert len(output.segments) == 2


# ==============================================================================
# Test Code Template
# ==============================================================================

class TestCodeTemplate:
    def test_validation(self):
        template = CodeTemplate()
        assert template.validate({"code": "print('hello')"})[0] is True
        assert template.validate({"instruction": "Write code"})[0] is False
    
    def test_format_basic(self):
        template = CodeTemplate()
        data = {"code": "def hello():\n    print('Hello')"}
        output = template.format(data)
        assert "def hello():" in output.text
        assert output.segments[-1]["compute_loss"] is True
    
    def test_format_with_language(self):
        template = CodeTemplate()
        data = {"code": "console.log('hi')", "language": "javascript"}
        output = template.format(data)
        assert "```javascript" in output.text
        assert output.metadata["language"] == "javascript"
    
    def test_format_with_context(self):
        template = CodeTemplate()
        data = {"code": "x = 1", "context": "import math"}
        output = template.format(data)
        assert "Context:" in output.text
        assert "import math" in output.text


# ==============================================================================
# Test Preference Template
# ==============================================================================

class TestPreferenceTemplate:
    def test_validation(self):
        template = PreferenceTemplate()
        assert template.validate({"prompt": "Q", "chosen": "A1"})[0] is True
        assert template.validate({"prompt": "Q", "rejected": "A2"})[0] is True
        assert template.validate({"prompt": "Q"})[0] is False
    
    def test_format_with_chosen(self):
        template = PreferenceTemplate()
        data = {"prompt": "Which is better?", "chosen": "Option A", "rejected": "Option B"}
        output = template.format(data)
        assert "Which is better?" in output.text
        assert "Option A" in output.text
        assert output.metadata["has_chosen"] is True
        assert output.metadata["rejected"] == "Option B"


# ==============================================================================
# Test Template Registry
# ==============================================================================

class TestTemplateRegistry:
    def test_list_templates(self):
        templates = TemplateRegistry.list_templates()
        assert "chat" in templates
        assert "instruction" in templates
        assert "qa" in templates
        assert "completion" in templates
        assert "dialogue" in templates
        assert "code" in templates
        assert "preference" in templates
    
    def test_get_class(self):
        cls = TemplateRegistry.get_class("chat")
        assert cls == ChatTemplate
    
    def test_get_class_not_found(self):
        with pytest.raises(KeyError):
            TemplateRegistry.get_class("nonexistent")
    
    def test_create(self):
        template = TemplateRegistry.create("instruction")
        assert isinstance(template, InstructionTemplate)
    
    def test_get_or_create_caches(self):
        t1 = TemplateRegistry.get_or_create("qa")
        t2 = TemplateRegistry.get_or_create("qa")
        # Should be cached
        assert t1 is t2


class TestGetTemplate:
    def test_get_template(self):
        template = get_template("chat")
        assert isinstance(template, ChatTemplate)
    
    def test_get_template_with_kwargs(self):
        template = get_template("chat", train_on_input=True)
        assert template.train_on_input is True


# ==============================================================================
# Test Dataset Integration
# ==============================================================================

class TestTemplatedDataset:
    def test_basic_iteration(self):
        data = [
            {"messages": [{"role": "user", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Hello"}]},
        ]
        template = ChatTemplate()
        dataset = TemplatedDataset(data, template)
        
        assert len(dataset) == 2
        
        outputs = list(dataset)
        assert len(outputs) == 2
        assert all(isinstance(o, TemplateOutput) for o in outputs)
    
    def test_with_tokenizer(self):
        data = [{"messages": [{"role": "user", "content": "Test"}]}]
        template = ChatTemplate()
        tokenizer = MockTokenizer()
        dataset = TemplatedDataset(data, template, tokenizer)
        
        output = dataset[0]
        assert isinstance(output, TokenizedOutput)


class TestApplyTemplateToBatch:
    def test_batch_without_tokenizer(self):
        batch = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]
        template = QATemplate()
        result = apply_template_to_batch(batch, template)
        assert "outputs" in result
        assert len(result["outputs"]) == 2
    
    def test_batch_with_tokenizer(self):
        batch = [
            {"messages": [{"role": "user", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Hello world"}]},
        ]
        template = ChatTemplate()
        tokenizer = MockTokenizer()
        result = apply_template_to_batch(batch, template, tokenizer)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "loss_mask" in result
        assert len(result["input_ids"]) == 2
        # Should be padded to same length
        assert len(result["input_ids"][0]) == len(result["input_ids"][1])


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    def test_full_pipeline_chat(self):
        """Test complete pipeline from data to tokenized output."""
        template = get_template("chat")
        tokenizer = MockTokenizer()
        
        data = {
            "system_prompt": "Be helpful.",
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
            ]
        }
        
        output = template(data, tokenizer)
        
        assert isinstance(output, TokenizedOutput)
        assert len(output.input_ids) > 0
        assert len(output.attention_mask) == len(output.input_ids)
        assert len(output.loss_mask) == len(output.input_ids)
        assert output.metadata["template"] == "chat"
        assert output.metadata["num_turns"] == 2
    
    def test_full_pipeline_instruction(self):
        """Test instruction template pipeline."""
        template = get_template("instruction")
        tokenizer = MockTokenizer()
        
        data = {
            "instruction": "Summarize the following text.",
            "input": "This is a long document about technology.",
            "output": "A document about technology.",
        }
        
        output = template(data, tokenizer)
        
        assert isinstance(output, TokenizedOutput)
        assert output.metadata["has_input"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
