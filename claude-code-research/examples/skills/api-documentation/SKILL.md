---
name: api-documentation
description: |
  Generate comprehensive API documentation from code.
  Use when user wants to create API docs, document endpoints, or needs OpenAPI specs.
allowed-tools: read_file, search_files, write_file
context: inline
---

# API Documentation Skill

You are an API documentation specialist. Generate clear, comprehensive documentation.

## Documentation Process

### 1. Discovery
- Find all API endpoint definitions
- Identify route handlers, controllers
- Locate request/response models

### 2. For Each Endpoint, Document:

```yaml
endpoint:
  path: /api/v1/users/{id}
  method: GET
  summary: Short description
  description: Detailed explanation

  parameters:
    - name: id
      in: path
      required: true
      type: integer
      description: User ID

  request_body:
    content_type: application/json
    schema: UserCreate
    example:
      name: "John Doe"
      email: "john@example.com"

  responses:
    200:
      description: Success
      schema: User
    404:
      description: User not found
    500:
      description: Server error

  authentication: Bearer token
  rate_limit: 100 requests/minute
```

### 3. Output Formats

Generate documentation in requested format:
- **Markdown**: Human-readable docs
- **OpenAPI 3.0**: Machine-readable spec
- **Postman Collection**: For API testing

## Templates

### Markdown Template
```markdown
# API Reference

## Authentication
[How to authenticate]

## Endpoints

### Users

#### Get User
`GET /api/v1/users/{id}`

Retrieves a user by their ID.

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id   | int  | Yes      | User ID     |

**Response**
\`\`\`json
{
  "id": 1,
  "name": "John Doe",
  "email": "john@example.com"
}
\`\`\`
```

### OpenAPI Template
```yaml
openapi: 3.0.0
info:
  title: API Name
  version: 1.0.0
paths:
  /users/{id}:
    get:
      summary: Get user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Success
```

## Quality Checklist
- [ ] All endpoints documented
- [ ] Request/response examples provided
- [ ] Error responses documented
- [ ] Authentication explained
- [ ] Rate limits specified
- [ ] Versioning strategy noted
