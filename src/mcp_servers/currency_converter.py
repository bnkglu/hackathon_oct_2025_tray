"""
This is an implementation of an MCP server that provides currency conversion
"""

from mcp.server.fastmcp import FastMCP
from src.util.utils import get_root_dir
import json

# Load the official currency rates (from 26 September 2025)
with open(get_root_dir() / 'data' / 'currency_rates.json', 'r') as f:
    currency_rates = json.load(f)

# Load the currency names
with open(get_root_dir() / 'data' / 'currency_names.json', 'r') as f:
    currency_names = json.load(f)
currency_names.update({"EUR": "Euro"}) # add Euro manually as it's not in the list

# TODO: implement your MCP server here

@mcp.tool()
def convert_currency():
    # TODO: implement this function using currency_rates
    pass

@mcp.resource("available_currencies")
def get_available_currencies():
    # TODO: implement this function using currency_names
    pass

if __name__ == "__main__":
    # TODO: Initialize and run the server
    pass
