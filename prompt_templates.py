from jinja2 import Template

SYS_MARKETING = "You are a sharp marketing copywriter. Be concise, on-brand, and punchy."
SYS_LEGAL = "You are a pragmatic legal drafter. Be clear, neutral, and minimal."
SYS_TECHDOC = "You are a precise technical writer. Be structured and accurate."

def system_for(domain: str) -> str:
    return {
        "marketing": SYS_MARKETING,
        "legal": SYS_LEGAL,
        "techdoc": SYS_TECHDOC,
    }.get(domain, "You are helpful and concise.")

TEMPLATE = """<s>[SYSTEM]
{{system}}
[/SYSTEM]
[INSTRUCTION]
{{instruction}}
[/INSTRUCTION]
[INPUT]
{{input}}
[/INPUT]
[RESPONSE]
"""

def format_example(domain, instruction, input_text):

    return Template(TEMPLATE).render(
        system=system_for(domain),
        instruction=instruction,
        input=input_text or "",
    )
