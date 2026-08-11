with open("models_extra.py") as f:
    content = f.read()

old_init = '''    def __init__(self, model, base_url, api_key=None, max_tokens=250,
                 seconds_per_query=0.0, max_retries=5, temperature=0.0):'''
new_init = '''    def __init__(self, model, base_url, api_key=None, max_tokens=None,
                 seconds_per_query=0.0, max_retries=5, temperature=None):'''

if old_init not in content:
    print("ERROR: __init__ signature not found as expected -- no changes made to init.")
else:
    content = content.replace(old_init, new_init, 1)
    print("Patched __init__ defaults (max_tokens=None, temperature=None).")

old_request = '''    def request_chat_model(self, msgs):
        messages = [{"role": role, "content": content} for content, role in msgs]
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )'''

new_request = '''    def request_chat_model(self, msgs):
        messages = [{"role": role, "content": content} for content, role in msgs]
        kwargs = {"model": self.model, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return self.client.chat.completions.create(**kwargs)'''

if old_request not in content:
    print("ERROR: request_chat_model body not found as expected -- no changes made to request method.")
else:
    content = content.replace(old_request, new_request, 1)
    print("Patched request_chat_model to conditionally omit max_tokens/temperature.")

with open("models_extra.py", "w") as f:
    f.write(content)
print("Done. Wrote updated models_extra.py")
