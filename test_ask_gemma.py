"""Tests for ask_gemma.py — covers geolocation, system prompt, and chat loop."""
import io
import json
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an ollama chat response."""
    resp = MagicMock()
    resp.message.content = content
    return resp


# ---------------------------------------------------------------------------
# get_location_info
# ---------------------------------------------------------------------------

class TestGetLocationInfo(unittest.TestCase):

    def _mock_urlopen(self, payload: dict):
        """Return a context-manager mock that yields JSON bytes."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.read = MagicMock(return_value=json.dumps(payload).encode())
        return cm

    @patch("urllib.request.urlopen")
    def test_returns_location_and_timezone(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen({
            "city": "Denver", "region": "Colorado", "country": "US",
            "timezone": "America/Denver",
        })
        import ask_gemma
        location, timezone = ask_gemma.get_location_info()
        self.assertEqual(location, "Denver, Colorado, US")
        self.assertEqual(timezone, "America/Denver")

    @patch("urllib.request.urlopen")
    def test_missing_timezone_falls_back(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen({
            "city": "Denver", "region": "Colorado", "country": "US",
        })
        import ask_gemma
        location, timezone = ask_gemma.get_location_info()
        self.assertEqual(timezone, ask_gemma._FALLBACK_TIMEZONE)

    @patch("urllib.request.urlopen")
    def test_empty_location_fields_fall_back(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen({"timezone": "America/Denver"})
        import ask_gemma
        location, _ = ask_gemma.get_location_info()
        self.assertEqual(location, ask_gemma._FALLBACK_LOCATION)

    @patch("urllib.request.urlopen", side_effect=OSError("network error"))
    def test_network_error_returns_fallbacks(self, _):
        import ask_gemma
        location, timezone = ask_gemma.get_location_info()
        self.assertEqual(location, ask_gemma._FALLBACK_LOCATION)
        self.assertEqual(timezone, ask_gemma._FALLBACK_TIMEZONE)


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------

class TestGetSystemPrompt(unittest.TestCase):

    @patch("ask_gemma.get_location_info", return_value=("Boulder, Colorado, US", "America/Denver"))
    def test_prompt_contains_location(self, _):
        import ask_gemma
        prompt = ask_gemma.get_system_prompt()
        self.assertIn("Boulder, Colorado, US", prompt)

    @patch("ask_gemma.get_location_info", return_value=("Boulder, Colorado, US", "America/Denver"))
    def test_prompt_contains_date_and_time(self, _):
        import ask_gemma
        prompt = ask_gemma.get_system_prompt()
        # Should contain a weekday name and a year
        import datetime
        self.assertIn(str(datetime.datetime.now().year), prompt)
        self.assertIn("Today is", prompt)
        self.assertIn("Current time is", prompt)

    @patch("ask_gemma.get_location_info", return_value=("Boulder, Colorado, US", "America/Denver"))
    def test_prompt_contains_assistant_instruction(self, _):
        import ask_gemma
        prompt = ask_gemma.get_system_prompt()
        self.assertIn("helpful assistant", prompt)


# ---------------------------------------------------------------------------
# chat_loop
# ---------------------------------------------------------------------------

class TestChatLoop(unittest.TestCase):

    def _run_loop(self, inputs: list[str], reply: str = "Hello!"):
        """
        Run chat_loop() with a scripted sequence of user inputs and a fixed
        model reply. Returns captured stdout.
        """
        import ask_gemma
        with patch("ask_gemma.get_location_info",
                   return_value=("Superior, Colorado, USA", "America/Denver")), \
             patch("ask_gemma.get_system_prompt",
                   return_value="You are a helpful assistant."), \
             patch("ask_gemma.chat", return_value=_make_chat_response(reply)) as mock_chat, \
             patch("builtins.input", side_effect=inputs), \
             patch("builtins.print") as mock_print:
            ask_gemma.chat_loop()
        return mock_chat, mock_print

    def test_session_header_printed(self):
        _, mock_print = self._run_loop(["exit"])
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertIn("Superior, Colorado, USA", printed)

    def test_exit_command_ends_loop(self):
        mock_chat, _ = self._run_loop(["exit"])
        mock_chat.assert_not_called()

    def test_quit_command_ends_loop(self):
        mock_chat, _ = self._run_loop(["quit"])
        mock_chat.assert_not_called()

    def test_empty_input_is_skipped(self):
        mock_chat, _ = self._run_loop(["", "exit"])
        mock_chat.assert_not_called()

    def test_message_sent_to_model(self):
        mock_chat, _ = self._run_loop(["Hello", "exit"])
        mock_chat.assert_called_once()
        # The messages list is mutated in-place after the call (assistant reply
        # is appended), so the user message is at [-2], not [-1].
        messages = mock_chat.call_args.kwargs["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0]["content"], "Hello")

    def test_reply_appended_to_history(self):
        import ask_gemma
        captured_messages = []

        def fake_chat(model, messages):
            captured_messages.extend(messages)
            return _make_chat_response("Hi there!")

        with patch("ask_gemma.get_location_info",
                   return_value=("Superior, Colorado, USA", "America/Denver")), \
             patch("ask_gemma.get_system_prompt",
                   return_value="You are a helpful assistant."), \
             patch("ask_gemma.chat", side_effect=fake_chat), \
             patch("builtins.input", side_effect=["Hello", "exit"]), \
             patch("builtins.print"):
            ask_gemma.chat_loop()

        roles = [m["role"] for m in captured_messages]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    def test_model_reply_is_printed(self):
        _, mock_print = self._run_loop(["Hi", "exit"], reply="Hey there!")
        printed = " ".join(str(c) for call in mock_print.call_args_list for c in call.args)
        self.assertIn("Hey there!", printed)

    def test_conversation_history_grows_across_turns(self):
        import ask_gemma
        call_args_list = []

        def fake_chat(model, messages):
            call_args_list.append(list(messages))
            return _make_chat_response("response")

        with patch("ask_gemma.get_location_info",
                   return_value=("Superior, Colorado, USA", "America/Denver")), \
             patch("ask_gemma.get_system_prompt",
                   return_value="You are a helpful assistant."), \
             patch("ask_gemma.chat", side_effect=fake_chat), \
             patch("builtins.input", side_effect=["First", "Second", "exit"]), \
             patch("builtins.print"):
            ask_gemma.chat_loop()

        # Second call should have more messages than the first
        self.assertGreater(len(call_args_list[1]), len(call_args_list[0]))

    def test_keyboard_interrupt_exits_gracefully(self):
        import ask_gemma
        with patch("ask_gemma.get_location_info",
                   return_value=("Superior, Colorado, USA", "America/Denver")), \
             patch("ask_gemma.get_system_prompt",
                   return_value="You are a helpful assistant."), \
             patch("ask_gemma.chat"), \
             patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("builtins.print"):
            # Should not raise
            ask_gemma.chat_loop()

    def test_eof_exits_gracefully(self):
        import ask_gemma
        with patch("ask_gemma.get_location_info",
                   return_value=("Superior, Colorado, USA", "America/Denver")), \
             patch("ask_gemma.get_system_prompt",
                   return_value="You are a helpful assistant."), \
             patch("ask_gemma.chat"), \
             patch("builtins.input", side_effect=EOFError), \
             patch("builtins.print"):
            ask_gemma.chat_loop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
