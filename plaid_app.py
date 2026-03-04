from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import plaid
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
import datetime
import os

#pip install flask flask-cors plaid-python

app = Flask(__name__, static_folder='.')
CORS(app)

# ----------------------------------------------------------------
# CONFIGURATION — replace with your Plaid dashboard credentials
# ----------------------------------------------------------------
PLAID_CLIENT_ID = ""
PLAID_SECRET    = ""   # swap for production secret later
PLAID_ENV       = "sandbox"               # change to "production" when approved
# ----------------------------------------------------------------

env_map = {
    "sandbox":    plaid.Environment.Sandbox,
    "development": plaid.Environment.Development,
    "production": plaid.Environment.Production,
}

configuration = plaid.Configuration(
    host=env_map[PLAID_ENV],
    api_key={"clientId": PLAID_CLIENT_ID, "secret": PLAID_SECRET},
)
api_client = plaid.ApiClient(configuration)
client     = plaid_api.PlaidApi(api_client)

# In-memory store (replace with a database in production)
access_tokens = {}


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/create_link_token", methods=["POST"])
def create_link_token():
    """Create a Plaid Link token to initialise the Link widget."""
    try:
        req = LinkTokenCreateRequest(
            products=[Products("transactions")],
            client_name="My Budget App",
            country_codes=[CountryCode("US")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id="local-user"),
        )
        response = client.link_token_create(req)
        return jsonify({"link_token": response["link_token"]})
    except plaid.ApiException as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/exchange_public_token", methods=["POST"])
def exchange_public_token():
    """Exchange the short-lived public token for a permanent access token."""
    try:
        public_token = request.json["public_token"]
        req      = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = client.item_public_token_exchange(req)
        access_token = response["access_token"]
        item_id      = response["item_id"]
        # Store against item_id (use a DB in production)
        access_tokens[item_id] = access_token
        return jsonify({"item_id": item_id, "success": True})
    except plaid.ApiException as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    """Return the last 90 days of transactions for all linked accounts."""
    if not access_tokens:
        return jsonify({"error": "No linked accounts. Please connect a bank first."}), 400
    try:
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=90)
        all_transactions = []
        for item_id, access_token in access_tokens.items():
            req = TransactionsGetRequest(
                access_token=access_token,
                start_date=start,
                end_date=end,
                options=TransactionsGetRequestOptions(count=500),
            )
            resp = client.transactions_get(req)
            for t in resp["transactions"]:
                all_transactions.append({
                    "date":     str(t["date"]),
                    "name":     t["name"],
                    "amount":   t["amount"],
                    "category": t["category"][0] if t["category"] else "Uncategorized",
                    "account":  t["account_id"],
                })
        return jsonify({"transactions": all_transactions})
    except plaid.ApiException as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/accounts", methods=["GET"])
def get_accounts():
    """Return balances for all linked accounts."""
    if not access_tokens:
        return jsonify({"error": "No linked accounts."}), 400
    try:
        all_accounts = []
        for item_id, access_token in access_tokens.items():
            req  = AccountsGetRequest(access_token=access_token)
            resp = client.accounts_get(req)
            for a in resp["accounts"]:
                all_accounts.append({
                    "name":     a["name"],
                    "type":     str(a["type"]),
                    "balance":  a["balances"]["current"],
                    "currency": a["balances"]["iso_currency_code"],
                })
        return jsonify({"accounts": all_accounts})
    except plaid.ApiException as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)