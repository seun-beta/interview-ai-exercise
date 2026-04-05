PROMPT_TEMPLATE = """Please answer the question based on the context below
Do not use your own knowledge, only use the information provided in the context.

Context:
{context}

Question: {query}

Answer:"""

QUERY_REWRITE_PROMPT = """\
You are a search query optimizer for the StackOne unified API documentation.

The documentation is stored as natural language descriptions of API endpoints in this format:
- "{{API_NAME}} API - {{Summary}}" (e.g. "HRIS API - List Employees")
- "{{HTTP_METHOD}} {{path}}" (e.g. "GET /unified/hris/employees")
- "Operation ID: {{operation_id}}" (e.g. "hris_list_employees")
- Field names like "first_name, last_name, work_email, job_title"
- Nested field descriptions like "employments contains: job_title, pay_rate"

The available APIs are: StackOne (connect sessions, accounts, connectors), \
HRIS (companies, employees, employments, documents, groups, jobs, locations, benefits, time off), \
ATS (applications, candidates, interviews, jobs, offers, lists, scorecards), \
LMS (courses, content, assignments, completions, users), \
IAM (users, roles, groups, policies), \
CRM (contacts, accounts, lists), \
Marketing (email templates, SMS templates, campaigns, content blocks).

Examples:
- "how do I get a list of workers?" -> "HRIS API List Employees GET /unified/hris/employees"
- "create a new job applicant" -> "ATS API Create Application POST /unified/ats/applications candidate"
- "what fields can I get for an employee?" -> "HRIS API List Employees GET /unified/hris/employees Employee fields first_name last_name work_email job_title department"
- "how to update a contact in the CRM?" -> "CRM API Update Contact PATCH /unified/crm/contacts contact first_name last_name emails phone_numbers"
- "get training courses" -> "LMS API List Courses GET /unified/lms/courses course learning content"
- "check what roles a user has" -> "IAM API Get User GET /unified/iam/users roles groups permissions"

Rewrite the following question to match this documentation vocabulary. \
Output only the rewritten query, nothing else.

Question: {query}"""
