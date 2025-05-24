TICKET_RESOLUTION_TEMPLATE = """You are a helpful banking support agent. Based on the following similar resolved tickets, provide a detailed response for the current issue.

Current Issue:
Title: {query_title}
Description: {query_description}

Similar Resolved Tickets:
{context}

Please provide a solution following these guidelines:
1. Start with a clear understanding of the issue
2. Provide step-by-step resolution steps

Your response:""" 



#2. Provide step-by-step resolution steps
#3. Include any relevant preventive measures
#4. Mention any potential follow-up actions if needed