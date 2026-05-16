from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from werkzeug.utils import secure_filename 
import os
from flask import Flask, abort, render_template, redirect, url_for, request, Response, session, flash
import csv
from io import StringIO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "global_xgb.pkl"
DATA_PATH = BASE_DIR / "all_transactions.csv"
METRICS_PATH = BASE_DIR / "global_metrics.json"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

LABEL_COL = "label"
DROP_COLS = ["label", "txn_id", "dt"]

app = Flask(__name__)
app.secret_key = "smartaml_prototype_secret_key"

#Global in-memory stores
MODEL = None
RAW_DF: pd.DataFrame | None = None
ENCODED_DF: pd.DataFrame | None = None
APP_DF: pd.DataFrame | None = None
CASE_DECISIONS = {}

USERS = {
    "tester1": "SmartAML2025",
    "tester2": "SmartAML2025",
    "tester3": "SmartAML2025",
    "tester4": "SmartAML2025",
    "tester5": "SmartAML2025",
    "tester6": "SmartAML2025",
}

def login_required():
    if "username" not in session:
        return redirect(url_for("index"))
    return None


def get_case_decision(txn_id: str) -> str:
    record = CASE_DECISIONS.get(str(txn_id))
    if isinstance(record, dict):
        return record.get("decision", "Pending Review")
    return record or "Pending Review"


def get_case_reviewer(txn_id: str) -> str:
    record = CASE_DECISIONS.get(str(txn_id))

    if isinstance(record, dict):
        reviewer = record.get("reviewed_by", "")

        if pd.isna(reviewer) or str(reviewer).lower() == "nan":
            return ""

        return str(reviewer)

    return ""


def get_available_datasets():
    datasets = [{"label": "Jan 2025", "value": "default"}]

    for file in sorted(UPLOAD_DIR.glob("*.csv")):
        name = file.stem.replace("_transactions", "").replace("_", " ").title()
        datasets.append({
            "label": name,
            "value": file.name
        })

    return datasets


def load_selected_dataset(selected_dataset):
    if selected_dataset and selected_dataset != "default":
        selected_path = UPLOAD_DIR / selected_dataset
        if selected_path.exists():
            df = pd.read_csv(selected_path)
        else:
            df = APP_DF.copy()
    else:
        df = APP_DF.copy()

    # Safety columns for uploaded files
    if "risk_label" not in df.columns:
        df["risk_label"] = "LOW"

    if "risk_score" not in df.columns:
        df["risk_score"] = 0.0

    if "country_dest" not in df.columns and "country_origin" in df.columns:
        df["country_dest"] = df["country_origin"]

    return df


if os.path.exists("case_decisions.csv"):
    df_decisions = pd.read_csv("case_decisions.csv", encoding="latin1")

    for _, row in df_decisions.iterrows():
        CASE_DECISIONS[str(row.get("txn_id", ""))] = {
            "decision": row.get("decision", "Pending Review"),
            "reviewed_by": row.get("reviewed_by", ""),
        }


def encode_like_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors your training-time encoding:
    object/category columns -> categorical codes.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def risk_label_from_score(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"

def map_product_type(value: str, channel: str = "") -> str:
    value = str(value).upper()
    channel = str(channel).upper()

    #Override logic first
    if channel == "CASH_OUT":
        return "Cash Withdrawal"

    if channel == "CASH_IN":
        return "Cash Deposit"

    mapping = {
        "TRANSFER_DOM": "Local Bank Transfer",
        "TRANSFER_INTL": "International Bank Transfer",
        "CARD": "Card Payment / Purchase",
        "LOAN_AUTO": "Loan / Auto Finance Payment",
        "ACCOUNT_EVENT": "Account Activity"
    }

    return mapping.get(value, value)


def map_channel(value: str, product_type: str = "") -> str:
    value = str(value).upper()
    product_type = str(product_type).upper()

    #special case for loan
    if "LOAN" in product_type:
        return "Loan Repayment Channel"

    mapping = {
        "TRANSFER": "Bank Transfer",
        "CASH_OUT": "Cash Withdrawal / Cash Outflow",
        "CASH_IN": "Cash Deposit / Cash Inflow",
        "ATM": "ATM Transaction",
        "CARD": "Card / Online Payment",
        "BANK_TRANSFER": "Bank Transfer",
        "ONLINE": "Online Banking",
        "MOBILE": "Mobile Banking",
        "BRANCH": "Branch / Counter Transaction"
    }

    return mapping.get(value, value.replace("_", " ").title())

FIRST_NAMES = [
    "Adam", "Aisyah", "Daniel", "Farah", "Hakim",
    "Nadia", "Irfan", "Sofia", "Rayyan", "Mira"
]

LAST_NAMES = [
    "Rahman", "Lim", "Tan", "Jamal", "Yusof",
    "Wong", "Hassan", "Lee", "Abdullah", "Salleh"
]


def generate_customer_name(customer_id: str) -> str:
    customer_id = str(customer_id)
    seed = sum(ord(ch) for ch in customer_id)

    first = FIRST_NAMES[seed % len(FIRST_NAMES)]
    last = LAST_NAMES[(seed // len(FIRST_NAMES)) % len(LAST_NAMES)]

    return f"{first} {last}"

BANKS = [
    "Baidara Bank",
    "Borneo Islamic Bank",
    "MayBunga Bank",
    "UOBorneo",
    "Standard Charcoal Bank",
    "Rimba Finance",
    "Darussalam Trust Bank",
]

def pick_bank(value: str) -> str:
    value = str(value)
    index = sum(ord(ch) for ch in value) % len(BANKS)
    return BANKS[index]


def mask_account(value: str, prefix: str = "ACC") -> str:
    value = str(value)
    last_digits = "".join(ch for ch in value if ch.isdigit())[-6:]
    if not last_digits:
        last_digits = value[-6:]
    return f"{prefix}-{last_digits}"


def get_transfer_type(row: pd.Series) -> str:
    product_type = str(row.get("product_type", "")).upper()
    origin = str(row.get("country_origin", ""))
    dest = str(row.get("country_dest", ""))

    if "TRANSFER_INTL" in product_type or origin != dest:
        return "International / Cross-Border Transfer"

    if "TRANSFER_DOM" in product_type:
        return "Domestic Interbank Transfer"

    if "CARD" in product_type:
        return "Card Transaction"

    if "LOAN" in product_type:
        return "Loan / Auto Finance Event"

    if "ACCOUNT" in product_type:
        return "Account Event"

    return "General Transaction"


def make_ai_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []

    amount_base = float(row.get("amount_base", 0) or 0)
    risk_score = float(row.get("risk_score", 0) or 0)
    origin = str(row.get("country_origin", ""))
    dest = str(row.get("country_dest", ""))
    product_type = str(row.get("product_type", ""))
    transfer_type = get_transfer_type(row)

    new_counterparty_flag = row.get("new_counterparty_flag", False)
    sum_amt_prev3 = float(row.get("sum_amt_prev3", 0) or 0)
    sum_amt_prev10 = float(row.get("sum_amt_prev10", 0) or 0)
    cnt_prev3 = float(row.get("cnt_prev3", 0) or 0)
    hour = int(row.get("hour", 0) or 0)

    if "International" in transfer_type:
        reasons.append(f"Cross-border transaction detected from {origin} to {dest}.")
    elif "Domestic" in transfer_type:
        reasons.append("Domestic interbank transfer identified within Brunei/local banking network.")

    if amount_base >= 100000:
        reasons.append("High-value transaction exceeds normal retail monitoring threshold.")
    elif amount_base >= 5000:
        reasons.append("Transaction amount is higher than typical low-value customer activity.")

    if bool(new_counterparty_flag):
        reasons.append("First-time interaction with this counterparty was detected.")

    if sum_amt_prev3 > 10000 or sum_amt_prev10 > 20000:
        reasons.append("Recent customer activity shows elevated transaction value.")

    if cnt_prev3 >= 3:
        reasons.append("Multiple recent transactions were observed in a short period.")

    if hour <= 5 or hour >= 22:
        reasons.append("Transaction occurred outside typical banking activity hours.")

    if risk_score >= 0.80:
        reasons.append("Model score indicates a strong suspicious pattern requiring investigator review.")

    if not reasons:
        reasons.append("Model identified a suspicious combination of transaction features.")

    return reasons[:5]


def load_metrics() -> dict[str, Any]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # fallback to your known values
    return {
        "pr_auc": 0.349,
        "roc_auc": 0.913,
        "precision": 0.70,
        "recall": 0.23,
        "threshold": 0.9966,
    }


def prepare_app_data() -> None:
    global MODEL, RAW_DF, ENCODED_DF, APP_DF

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    MODEL = joblib.load(MODEL_PATH)

    raw_df = pd.read_csv(DATA_PATH, parse_dates=["dt"], low_memory=False)

    # Keep a manageable subset for the app demo
    # You can increase this later if needed
    sample_size = min(3000, len(raw_df))
    if len(raw_df) > sample_size:
        sampled = raw_df.sample(n=sample_size, random_state=42).copy()
    else:
        sampled = raw_df.copy()

    sampled.reset_index(drop=True, inplace=True)

    # Make sure txn_id exists and is string-like
    if "txn_id" not in sampled.columns:
        sampled["txn_id"] = [f"TX{i:05d}" for i in range(1, len(sampled) + 1)]
    sampled["txn_id"] = sampled["txn_id"].astype(str)

    encoded = encode_like_train(sampled)

    feature_df = encoded.drop(columns=[c for c in DROP_COLS if c in encoded.columns], errors="ignore")
    proba = MODEL.predict_proba(feature_df)[:, 1]

    sampled["risk_score"] = proba
    sampled["risk_label"] = sampled["risk_score"].apply(risk_label_from_score)

    # Sort by risk score descending for dashboard focus
    sampled.sort_values("risk_score", ascending=False, inplace=True)
    sampled.reset_index(drop=True, inplace=True)

    RAW_DF = sampled
    ENCODED_DF = encoded.loc[sampled.index] if len(encoded) == len(sampled) else encode_like_train(sampled)
    APP_DF = sampled


@app.before_request
def ensure_loaded() -> None:
    global APP_DF
    if APP_DF is None:
        prepare_app_data()


def get_dashboard_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert APP_DF is not None

    high_df = APP_DF[APP_DF["risk_label"] == "HIGH"].head(10).copy()

    recent_df = APP_DF.copy()
    if "dt" in recent_df.columns:
        recent_df = recent_df.sort_values("dt", ascending=False).head(10)
    else:
        recent_df = recent_df.head(10)

    def row_to_dict(row: pd.Series) -> dict[str, Any]:
        return {
            "txn_id": str(row.get("txn_id", "")),
            "amount": row.get("amount", ""),
            "amount_base": row.get("amount_base", ""),
            "country": row.get("country_dest", row.get("country_origin", "")),
            "risk_label": row.get("risk_label", "LOW"),
            "risk_score": round(float(row.get("risk_score", 0.0)), 4),
            "product_type": row.get("product_type", ""),
            "dt": str(row.get("dt", ""))[:19],
        }

    high_rows = [row_to_dict(r) for _, r in high_df.iterrows()]
    recent_rows = [row_to_dict(r) for _, r in recent_df.iterrows()]
    return high_rows, recent_rows


def get_risk_summary() -> dict[str, int]:
    assert APP_DF is not None

    counts = APP_DF["risk_label"].value_counts().to_dict()
    return {
        "high": int(counts.get("HIGH", 0)),
        "medium": int(counts.get("MEDIUM", 0)),
        "low": int(counts.get("LOW", 0)),
    }


def get_transaction_by_id(txn_id: str) -> dict[str, Any]:
    assert APP_DF is not None

    match = APP_DF[APP_DF["txn_id"].astype(str) == str(txn_id)]
    if match.empty:
        abort(404, description=f"Transaction {txn_id} not found")

    row = match.iloc[0]
    customer_id = str(row.get("customer_id", ""))
    customer_summary = get_customer_summary(customer_id, row.get("txn_id", ""))
    counterparty_id = str(row.get("counterparty_id", ""))
    account_id = str(row.get("account_id", ""))

    origin_bank = pick_bank(account_id or customer_id)
    destination_bank = pick_bank(counterparty_id)
    transfer_type = get_transfer_type(row)

    is_cash_transaction = "Cash" in map_product_type(
    row.get("product_type", ""),
    row.get("channel", "")
    )

    return {
        "txn_id": str(row.get("txn_id", "")),
        "customer_name": f"Customer {customer_id}",
        "counterparty_name": f"Counterparty {counterparty_id}",
        "origin_account": mask_account(account_id or customer_id, "ACC"),
        "destination_account": mask_account(counterparty_id, "CP"),
        "origin_bank": origin_bank,
        "destination_bank": destination_bank,
        "transfer_type": "Cash Transaction" if is_cash_transaction else transfer_type,
        "transfer_flow": "Brunei cash transaction" if is_cash_transaction else f"{row.get('country_origin', '')} → {row.get('country_dest', '')}",
        "amount": row.get("amount", ""),
        "amount_base": row.get("amount_base", ""),
        "country_origin": row.get("country_origin", ""),
        "country_dest": row.get("country_dest", ""),
        "channel": map_channel(
            row.get("channel", ""),
            row.get("product_type", "")
        ),
        "product_type": map_product_type(
            row.get("product_type", ""),
            row.get("channel", "")
        ),
        "raw_channel": row.get("channel", ""),
        "raw_product_type": row.get("product_type", ""),
        "currency": row.get("currency", ""),
        "dt": str(row.get("dt", ""))[:19],
        "risk_label": row.get("risk_label", "LOW"),
        "risk_score": round(float(row.get("risk_score", 0.0)), 4),
        "reasons": make_ai_reasons(row),
        "decision": get_case_decision(str(row.get("txn_id", ""))),
        "reviewed_by": get_case_reviewer(row.get("txn_id", "")),
        "locked": bool(get_case_reviewer(row.get("txn_id", ""))) and get_case_reviewer(row.get("txn_id", "")) != session.get("username"),
        "customer_summary": customer_summary,
        "is_cash_transaction": "Cash" in map_product_type(row.get("product_type", ""), row.get("channel", "")),
        "context_title": "Cash Transaction Context" if "Cash" in map_product_type(row.get("product_type", ""), row.get("channel", "")) else "Transfer Context",
    }


def get_customer_summary(customer_id: str, current_txn_id: str):
    assert APP_DF is not None

    cust_df = APP_DF[APP_DF["customer_id"].astype(str) == str(customer_id)].copy()

    total_transactions = len(cust_df)
    total_amount = round(float(cust_df["amount"].sum()), 2) if "amount" in cust_df.columns else 0
    avg_amount = round(float(cust_df["amount"].mean()), 2) if total_transactions > 0 else 0
    high_risk_count = int((cust_df["risk_label"] == "HIGH").sum()) if "risk_label" in cust_df.columns else 0

    recent_rows = []
    if "dt" in cust_df.columns:
        cust_df = cust_df.sort_values("dt", ascending=False)

    for _, row in cust_df.head(8).iterrows():
        txn_id = str(row.get("txn_id", ""))
        decision = get_case_decision(txn_id)

        recent_rows.append({
            "txn_id": txn_id,
            "dt": "Unknown" if str(row.get("dt", ""))[:10] == "1970-01-01" else str(row.get("dt", ""))[:10],
            "amount": row.get("amount", ""),
            "country": row.get("country_dest", row.get("country_origin", "")),
            "risk_label": row.get("risk_label", ""),
            "risk_score": round(float(row.get("risk_score", 0.0)), 4),
            "status": decision,
            "is_current": txn_id == str(current_txn_id),
        })

    return {
        "customer_name": generate_customer_name(customer_id),
        "customer_id": customer_id,
        "total_transactions": total_transactions,
        "total_amount": total_amount,
        "avg_amount": avg_amount,
        "high_risk_count": high_risk_count,
        "recent_rows": recent_rows,
    }


@app.route("/", methods=["GET", "POST"])
@app.route("/index.html", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username in USERS and USERS[username] == password:
            session["username"] = username
            return redirect(url_for("home"))

        return render_template("index.html", error="Invalid username or password")

    return render_template("index.html")


@app.route("/home")
@app.route("/home.html")
def home():
    guard = login_required()
    if guard:
        return guard
    
    return render_template("home.html")


@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard():
    guard = login_required()
    if guard:
        return guard
    
    assert APP_DF is not None

    selected_dataset = session.get("selected_dataset", "default")
    df = load_selected_dataset(selected_dataset).head(500)

    total = len(df)
    pending = 0
    confirmed = 0
    false_positive = 0

    for _, row in df.iterrows():
        txn_id = str(row.get("txn_id", ""))
        status = get_case_decision(txn_id)

        if status == "Confirmed Suspicious":
            confirmed += 1
        elif status == "False Positive":
            false_positive += 1
        else:
            pending += 1

    high_rows = []
    high_df = df[df["risk_label"] == "HIGH"].head(5)

    for _, row in high_df.iterrows():
        high_rows.append({
            "customer_name": generate_customer_name(row.get("customer_id", "")),
            "status": get_case_decision(str(row.get("txn_id", ""))),
            "txn_id": str(row.get("txn_id", "")),
            "amount": row.get("amount", ""),
            "country": row.get("country_dest", row.get("country_origin", "")),
            "risk_label": row.get("risk_label", ""),
            "risk_score": round(float(row.get("risk_score", 0.0)), 4),
        })

    recent_rows = []
    recent_df = df.sort_values("dt", ascending=False).head(5) if "dt" in df.columns else df.head(5)

    for _, row in recent_df.iterrows():
        txn_id = str(row.get("txn_id", ""))
        manual_decision = get_case_decision(str(row.get("txn_id", "")))

        if manual_decision == "Confirmed Suspicious":
            status = "Confirmed Suspicious"
        elif manual_decision == "False Positive":
            status = "False Positive"
        else:
            status = "Pending Review"

        recent_rows.append({
            "customer_name": generate_customer_name(row.get("customer_id", "")),
            "txn_id": txn_id,
            "amount": row.get("amount", ""),
            "country": row.get("country_dest", row.get("country_origin", "")),
            "risk_label": row.get("risk_label", ""),
            "risk_score": round(float(row.get("risk_score", 0.0)), 4),
            "status": status,
        })

    return render_template(
        "dashboard.html",
        total=total,
        pending=pending,
        confirmed=confirmed,
        false_positive=false_positive,
        high_rows=high_rows,
        recent_rows=recent_rows
    )


@app.route("/investigation")
@app.route("/investigation.html")
def investigation_default():
    guard = login_required()
    if guard:
        return guard
    
    # Default to top high-risk transaction
    assert APP_DF is not None
    top_txn_id = str(APP_DF.iloc[0]["txn_id"])
    tx = get_transaction_by_id(top_txn_id)
    return render_template("investigation.html", tx=tx)


@app.route("/decision/<txn_id>", methods=["POST"])
def decision(txn_id: str):
    decision = request.form.get("decision")

    current_user = session.get("username", "unknown")
    existing = CASE_DECISIONS.get(str(txn_id))

    if isinstance(existing, dict):
        existing_reviewer = existing.get("reviewed_by", "")

        if not pd.isna(existing_reviewer) and str(existing_reviewer).lower() != "nan" and existing_reviewer:
            return redirect(url_for("investigation", txn_id=txn_id))

    CASE_DECISIONS[str(txn_id)] = {
        "decision": decision,
        "reviewed_by": current_user,
    }

    df_save = pd.DataFrame([
        {
            "txn_id": key,
            "decision": value.get("decision", "Pending Review"),
            "reviewed_by": value.get("reviewed_by", ""),
        }
        for key, value in CASE_DECISIONS.items()
    ])
    
    df_save.to_csv("case_decisions.csv", index=False)

    return redirect(url_for("investigation", txn_id=txn_id))


@app.route("/investigation/<txn_id>")
def investigation(txn_id: str):
    tx = get_transaction_by_id(txn_id)
    return render_template("investigation.html", tx=tx)


@app.route("/reports")
@app.route("/reports.html")
def reports():
    guard = login_required()
    if guard:
        return guard
    
    assert APP_DF is not None

    selected_dataset = session.get("selected_dataset", "default")
    df = load_selected_dataset(selected_dataset)

    if "dt" in df.columns:
        df["dt_clean"] = pd.to_datetime(df["dt"], errors="coerce")
        valid_dates = df["dt_clean"].dropna()

        if not valid_dates.empty:
            default_start = valid_dates.min().strftime("%Y-%m-%d")
            default_end = valid_dates.max().strftime("%Y-%m-%d")
        else:
            default_start = "2025-01-01"
            default_end = "2025-01-31"
    else:
        default_start = "2025-01-01"
        default_end = "2025-01-31"

    start_date = request.args.get("start_date", default_start)
    end_date = request.args.get("end_date", default_end)

    if "dt" in df.columns:
        df["dt_clean"] = pd.to_datetime(df["dt"], errors="coerce")

        if start_date:
            df = df[df["dt_clean"] >= pd.to_datetime(start_date)]

        if end_date:
            df = df[df["dt_clean"] <= pd.to_datetime(end_date)]

    total_reviewed = len(df)
    suspicious = 0
    false_positive = 0
    pending = 0

    for _, row in df.iterrows():
        txn_id = str(row.get("txn_id", ""))
        decision = CASE_DECISIONS.get(txn_id)

        if decision == "Confirmed Suspicious":
            suspicious += 1
        elif decision == "False Positive":
            false_positive += 1
        else:
            pending += 1

    false_positive_rate = round((false_positive / total_reviewed) * 100, 2) if total_reviewed else 0
    suspicious_rate = round((suspicious / total_reviewed) * 100, 2) if total_reviewed else 0

    top_products = (
        df["product_type"]
        .apply(map_product_type)
        .value_counts()
        .head(5)
        .to_dict()
        if "product_type" in df.columns else {}
    )

    report = {
        "start_date": start_date,
        "end_date": end_date,
        "total_reviewed": total_reviewed,
        "suspicious": suspicious,
        "false_positive": false_positive,
        "pending": pending,
        "false_positive_rate": false_positive_rate,
        "suspicious_rate": suspicious_rate,
        "top_products": top_products,
    }

    return render_template("reports.html", report=report)

@app.route("/export_csv")
def export_csv():
    guard = login_required()
    if guard:
        return guard
    
    assert APP_DF is not None

    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")

    selected_dataset = session.get("selected_dataset", "default")
    df = load_selected_dataset(selected_dataset)

    if "dt" in df.columns:
        df["dt_clean"] = pd.to_datetime(df["dt"], errors="coerce")

        if start_date:
            df = df[df["dt_clean"] >= pd.to_datetime(start_date)]

        if end_date:
            df = df[df["dt_clean"] <= pd.to_datetime(end_date)]

    export_rows = []

    for _, row in df.iterrows():
        txn_id = str(row.get("txn_id", ""))
        decision = get_case_decision(txn_id)

        export_rows.append({
            "Alert Ref": txn_id,
            "Alert Date": "Unknown" if str(row.get("dt", ""))[:10] == "1970-01-01" else str(row.get("dt", ""))[:10],
            "Customer Name": generate_customer_name(row.get("customer_id", "")),
            "Customer ID": row.get("customer_id", ""),
            "Product Type": map_product_type(row.get("product_type", ""), row.get("channel", "")),
            "Channel": map_channel(row.get("channel", ""), row.get("product_type", "")),
            "Amount": row.get("amount", ""),
            "Country Origin": row.get("country_origin", ""),
            "Country Destination": row.get("country_dest", ""),
            "AI Risk": row.get("risk_label", ""),
            "Risk Score": round(float(row.get("risk_score", 0.0)), 4),
            "Investigation Status": decision,
        })

    output = StringIO()
    pd.DataFrame(export_rows).to_csv(output, index=False)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment;filename=smartaml_report.csv"
        }
    )

@app.route("/transactions")
@app.route("/transactions.html")
def transactions():
    guard = login_required()
    if guard:
        return guard
    
    assert APP_DF is not None

    filter_status = request.args.get("status", "All")
    search_query = request.args.get("search", "").strip()
    
    selected_dataset = request.args.get("dataset", session.get("selected_dataset", "default"))
    session["selected_dataset"] = selected_dataset
    tx_df = load_selected_dataset(selected_dataset).head(200)
    datasets = get_available_datasets()

    rows = []
    for _, row in tx_df.iterrows():
        risk = row.get("risk_label", "LOW")
        txn_id = str(row.get("txn_id", ""))

        status = get_case_decision(txn_id)

        display_country = row.get("country_dest", row.get("country_origin", ""))
        customer_name = generate_customer_name(row.get("customer_id", ""))

        if search_query:
            searchable_text = f"{txn_id} {customer_name} {row.get('customer_id', '')} {display_country}".lower()
            
            if search_query.lower() not in searchable_text:
                continue

        #FILTER LOGIC
        if filter_status != "All" and status != filter_status:
            continue

        rows.append({
            "txn_id": txn_id,
            "dt": "Unknown" if str(row.get("dt", ""))[:10] == "1970-01-01" else str(row.get("dt", ""))[:10],
            "customer_name": customer_name,
            "customer_id": row.get("customer_id", ""),
            "amount": row.get("amount", ""),
            "country": row.get("country_dest", row.get("country_origin", "")),
            "risk_label": risk,
            "risk_score": round(float(row.get("risk_score", 0.0)), 4),
            "status": status,
        })

    return render_template("transactions.html", rows=rows, filter_status=filter_status, search_query=search_query, datasets=datasets, selected_dataset=selected_dataset)

@app.route("/about")
@app.route("/about.html")
def about():
    guard = login_required()
    if guard:
        return guard
    
    return render_template("about.html")

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    guard = login_required()
    if guard:
        return guard

    uploaded_file = request.files.get("dataset")

    if not uploaded_file or uploaded_file.filename == "":
        return redirect(url_for("transactions"))

    filename = secure_filename(uploaded_file.filename)

    if not filename.endswith(".csv"):
        return redirect(url_for("transactions"))

    save_path = UPLOAD_DIR / filename
    uploaded_file.save(save_path)

    return redirect(url_for("transactions"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)