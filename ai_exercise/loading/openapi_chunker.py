import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _resolve_ref(ref: str, schemas: dict[str, Any]) -> dict[str, Any]:
    """Resolve a single $ref to its schema dict."""
    schema_name = ref.rsplit("/", 1)[-1]
    result: dict[str, Any] = schemas[schema_name]
    return result


def _get_response_fields(operation: dict[str, Any], schemas: dict[str, Any]) -> list[str]:
    """Follow the 200/201 response $ref one level and return field descriptions."""
    responses = operation["responses"]
    if "200" not in responses and "201" not in responses:
        return []
    status_code = "200" if "200" in responses else "201"
    response = responses[status_code]

    # Some endpoints (MCP, proxy) return no JSON body
    if "content" not in response or "application/json" not in response["content"]:
        return []

    schema_entry = response["content"]["application/json"]["schema"]

    # Some endpoints (accounts list, connectors, rpc) use inline schemas
    if "$ref" not in schema_entry:
        return []

    response_schema = _resolve_ref(schema_entry["$ref"], schemas)
    properties = response_schema.get("properties", {})

    # Most endpoints wrap the payload in a "data" key
    if "data" in properties:
        data_property = properties["data"]
        # Paginated list: data is an array with items.$ref
        if data_property.get("type") == "array":
            if "$ref" not in data_property.get("items", {}):
                return _describe_fields(properties)
            inner_schema = _resolve_ref(data_property["items"]["$ref"], schemas)
            return _describe_fields(inner_schema["properties"])
        # Single resource: data is a $ref (some use allOf/oneOf instead, fall through)
        if "$ref" in data_property:
            inner_schema = _resolve_ref(data_property["$ref"], schemas)
            return _describe_fields(inner_schema["properties"])

    # StackOne platform endpoints (ConnectSession, LinkedAccount, etc.) have no data wrapper
    return _describe_fields(properties)


def _describe_fields(properties: dict[str, Any]) -> list[str]:
    """Turn schema properties into 'name (type): description (default: X)' lines."""
    lines = []
    for name, prop in properties.items():
        field_type = prop.get("type", "")
        parts = [f"{name} ({field_type})" if field_type else name]
        desc = prop.get("description", "")
        if desc:
            parts.append(f": {desc}")
        default = prop.get("default")
        if default is not None:
            parts.append(f" (default: {default})")
        lines.append("".join(parts))
    return lines


def _get_request_fields(operation: dict[str, Any], schemas: dict[str, Any]) -> list[str]:
    """Follow the requestBody $ref one level and return field descriptions."""
    if "requestBody" not in operation:
        return []
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_schema = _resolve_ref(request_ref, schemas)
    return _describe_fields(request_schema["properties"])


def chunk_endpoints(spec: dict[str, Any]) -> list[Document]:
    """One document per endpoint. Grabs keys directly from the spec."""
    schemas = spec["components"]["schemas"]
    api_name = spec["info"]["title"]
    tag_descriptions = {tag["name"]: tag.get("description", "") for tag in spec.get("tags", [])}

    documents: list[Document] = []

    for path, path_item in spec["paths"].items():
        for method in ("get", "post", "put", "patch", "delete"):
            if method not in path_item:
                continue
            operation = path_item[method]
            tags = operation.get("tags", [])
            tag_description = tag_descriptions.get(tags[0], "") if tags else ""

            lines = [
                f"{api_name} API - {operation.get('summary', '')}",
                f"{method.upper()} {path}",
                f"Operation ID: {operation['operationId']}",
                "",
                operation.get("description", "") or operation.get("summary", ""),
                tag_description,
                "",
            ]

            for parameter in operation.get("parameters", []):
                lines.append(f"- {parameter['name']} ({parameter['in']}): {parameter.get('description', '')}")

            response_fields = _get_response_fields(operation, schemas)
            if response_fields:
                lines.append(f"\nResponse fields: {', '.join(response_fields)}")

            request_fields = _get_request_fields(operation, schemas)
            if request_fields:
                lines.append(f"\nRequest body fields: {', '.join(request_fields)}")

            documents.append(Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "paths",
                    "method": method.upper(),
                    "path": path,
                    "operation_id": operation["operationId"],
                    "category": api_name,
                    "tag": tags[0] if tags else "",
                },
            ))

    logger.info("Chunked %d endpoints from %s", len(documents), api_name)
    return documents


def chunk_supplementary(spec: dict[str, Any]) -> list[Document]:
    """One overview chunk per API."""
    api_name = spec["info"]["title"]
    auth_type = ", ".join(spec["components"]["securitySchemes"].keys())
    tag_descriptions = {tag["name"]: tag.get("description", "") for tag in spec.get("tags", [])}

    lines = [
        f"{api_name} API Overview",
        spec["info"].get("description", ""),
        f"Version: {spec['info']['version']}",
        f"Authentication: {auth_type}",
        "",
    ]
    for tag_name, tag_description in tag_descriptions.items():
        lines.append(f"- {tag_name}: {tag_description}" if tag_description else f"- {tag_name}")

    documents = [Document(
        page_content="\n".join(lines),
        metadata={"source": "info", "category": api_name},
    )]
    logger.info("Created %d supplementary chunks for %s", len(documents), api_name)
    return documents
