"""
EquiLens AI — MCP Routes

MCP-compliant endpoints for tool listing, tool invocation,
and resource exposure.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/tools")
async def list_tools():
    """
    List all available MCP tools.

    Returns a JSON array of tool definitions following the MCP spec.
    """
    # TODO: Return registered tool definitions
    return {"tools": []}


@router.post("/tools/{tool_name}/invoke")
async def invoke_tool(tool_name: str):
    """
    Invoke a specific MCP tool by name.

    Args:
        tool_name: The registered name of the tool to invoke.
    """
    # TODO: Dispatch to appropriate handler
    return {"status": "not_implemented", "tool": tool_name}


@router.get("/resources")
async def list_resources():
    """
    List all available MCP resources.

    Returns a JSON array of resource definitions.
    """
    # TODO: Return registered resource definitions
    return {"resources": []}
