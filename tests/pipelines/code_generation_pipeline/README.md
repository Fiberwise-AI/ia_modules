# Code Generation Pipeline

AI-powered code generation and editing pipeline using LLM to create and modify code.

## Features

- ✅ Read existing codebase
- ✅ Generate new code files
- ✅ Edit existing code
- ✅ Apply guardrails to prevent malicious code
- ✅ Execute hooks for monitoring
- ✅ Save generated code to external directory (NOT part of ia_modules)

## Usage

```bash
python run_code_generator.py "Create a FastAPI endpoint for user registration"
```

## Pipeline Steps

1. **Task Analysis** - Analyze the coding task
2. **Code Reading** - Read relevant existing code (if any)
3. **Code Generation** - Generate or edit code using LLM
4. **Code Validation** - Validate syntax and security
5. **File Writing** - Save to output directory

## Output

Generated code is saved to `generated_code/` directory, separate from ia_modules source code.
