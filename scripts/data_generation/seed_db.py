"""Seed pacify.db — mock operational database for PacifyIQ.

Deliberately positions orders against the policy boundaries defined in
canonical_facts.md so that eligibility logic is genuinely exercised.
"""
import os, sys
from pathlib import Path
# project-relative: <root>/data
DATA_ROOT = str(Path(__file__).resolve().parents[2] / "data")

import sqlite3, random, os
from datetime import datetime, timedelta
from faker import Faker

random.seed(42)
fake = Faker("en_IN")
Faker.seed(42)

DB = f"{DATA_ROOT}/db/pacify.db"
TODAY = datetime(2026, 8, 21)
os.makedirs(os.path.dirname(DB), exist_ok=True)
if os.path.exists(DB):
    os.remove(DB)

con = sqlite3.connect(DB)
c = con.cursor()

c.executescript("""
CREATE TABLE customers (
  customer_id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT,
  region TEXT, country TEXT, city TEXT, pincode TEXT,
  created_at TEXT, is_business INTEGER, total_orders INTEGER
);
CREATE TABLE products (
  sku TEXT PRIMARY KEY, name TEXT, category TEXT, brand TEXT,
  price REAL, warranty_months INTEGER, restocking_pct REAL,
  in_stock INTEGER, is_refurbished INTEGER
);
CREATE TABLE orders (
  order_id TEXT PRIMARY KEY, customer_id TEXT, sku TEXT, quantity INTEGER,
  unit_price REAL, discount REAL, shipping_charge REAL, total_paid REAL,
  payment_method TEXT, emi_tenure INTEGER, is_no_cost_emi INTEGER,
  order_date TEXT, dispatch_date TEXT, delivery_date TEXT,
  status TEXT, shipping_method TEXT, is_opened INTEGER,
  tracking_ref TEXT, region TEXT,
  FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY(sku) REFERENCES products(sku)
);
CREATE TABLE returns (
  return_id TEXT PRIMARY KEY, order_id TEXT, requested_date TEXT,
  reason TEXT, status TEXT, restocking_fee REAL, return_shipping REAL,
  refund_amount REAL, refund_status TEXT, refund_initiated_date TEXT
);
CREATE TABLE warranty_claims (
  claim_id TEXT PRIMARY KEY, order_id TEXT, raised_date TEXT,
  fault_description TEXT, error_code TEXT, status TEXT, outcome TEXT
);
CREATE TABLE tickets (
  ticket_id TEXT PRIMARY KEY, customer_id TEXT, order_id TEXT,
  created_at TEXT, intent TEXT, sentiment TEXT, priority TEXT,
  status TEXT, resolved_by TEXT, summary TEXT
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_tickets_customer ON tickets(customer_id);
""")

# ---------- products (canonical_facts.md S8) ----------
PRODUCTS = [
    ("PB14",     "Pacify ProBook 14",      "laptop",    "Pacify",    64900, 24, 0.10, 1, 0),
    ("PB16",     "Pacify ProBook 16",      "laptop",    "Pacify",    89900, 24, 0.10, 1, 0),
    ("PB14L",    "Pacify ProBook 14 Lite", "laptop",    "Pacify",    47900, 24, 0.10, 1, 0),
    ("PPX",      "Pacify Phone X",         "phone",     "Pacify",    38900, 24, 0.10, 1, 0),
    ("PPXP",     "Pacify Phone X Pro",     "phone",     "Pacify",    54900, 24, 0.10, 0, 0),
    ("PT11",     "Pacify Tab 11",          "tablet",    "Pacify",    29900, 24, 0.10, 1, 0),
    ("PV27",     "Pacify Vision 27",       "monitor",   "Pacify",    24900, 24, 0.10, 1, 0),
    ("PV32",     "Pacify Vision 32",       "monitor",   "Pacify",    41900, 24, 0.10, 1, 0),
    ("PKL",      "Pacify KeyLite",         "accessory", "Pacify",     3499,  6, 0.00, 1, 0),
    ("PSP",      "Pacify SoundPods",       "accessory", "Pacify",     5999,  6, 0.00, 1, 0),
    ("NWU15",    "Northwind Ultra 15",     "laptop",    "Northwind", 72900, 12, 0.10, 1, 0),
    ("KN9",      "Kestrel Note 9",         "phone",     "Kestrel",   31900, 12, 0.10, 1, 0),
    ("PB14R",    "Pacify ProBook 14 (Refurbished)", "laptop", "Pacify", 44900, 6, 0.10, 1, 1),
    ("PV27R",    "Pacify Vision 27 (Refurbished)",  "monitor","Pacify", 17900, 6, 0.10, 1, 1),
]
c.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", PRODUCTS)
SKUS = [p[0] for p in PRODUCTS]
PRICE = {p[0]: p[4] for p in PRODUCTS}
CAT = {p[0]: p[2] for p in PRODUCTS}

EU = ["Germany", "France", "Netherlands", "Ireland", "Spain", "Italy"]
IN_CITIES = ["Bengaluru", "Mumbai", "Delhi", "Pune", "Chennai", "Hyderabad",
             "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]

# ---------- customers ----------
customers = []
for i in range(500):
    cid = f"CUS-{10000+i}"
    is_eu = i % 10 == 0                      # 10% EU customers
    country = random.choice(EU) if is_eu else "India"
    customers.append((
        cid, fake.name(), f"user{i}@example.com", fake.msisdn()[:10],
        "EU" if is_eu else "IN", country,
        fake.city() if is_eu else random.choice(IN_CITIES),
        fake.postcode() if not is_eu else str(random.randint(10000, 99999)),
        (TODAY - timedelta(days=random.randint(30, 900))).strftime("%Y-%m-%d"),
        1 if i % 25 == 0 else 0, 0))
c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", customers)
CIDS = [x[0] for x in customers]
REGION = {x[0]: x[4] for x in customers}

PAY = ["upi", "credit_card", "debit_card", "net_banking", "emi", "cod"]
STATUSES = ["delivered"] * 62 + ["in_transit"] * 10 + ["processing"] * 6 + \
           ["dispatched"] * 8 + ["cancelled"] * 5 + ["returned"] * 5 + \
           ["refund_in_progress"] * 3 + ["delivery_failed"] * 1

orders = []
oid_n = 10000

def make_order(cid, sku, days_since_delivery=None, status=None, opened=None,
               payment=None, qty=1, no_cost_emi=0, force_id=None):
    """days_since_delivery=None means not yet delivered."""
    global oid_n
    oid = force_id or f"PAC-2026-{oid_n}"
    oid_n += 1
    price = PRICE[sku]
    disc = round(price * random.choice([0, 0, 0, 0.05, 0.08]), 0)
    paid_unit = price - disc
    subtotal = paid_unit * qty
    ship = 0 if subtotal >= 5000 else 99
    method = payment or random.choice(PAY)
    tenure = random.choice([3, 6, 9, 12]) if method == "emi" else None
    if method == "cod":
        ship += 50
    if days_since_delivery is not None:
        deliv = TODAY - timedelta(days=days_since_delivery)
        disp = deliv - timedelta(days=random.randint(3, 7))
        order_d = disp - timedelta(days=1)
        st = status or "delivered"
    else:
        order_d = TODAY - timedelta(days=random.randint(0, 6))
        disp = order_d + timedelta(days=1) if status in ("in_transit", "dispatched") else None
        deliv = None
        st = status or "processing"
    op = opened if opened is not None else (1 if random.random() < 0.7 else 0)
    orders.append((
        oid, cid, sku, qty, paid_unit, disc, ship, subtotal + ship, method,
        tenure, no_cost_emi,
        order_d.strftime("%Y-%m-%d"),
        disp.strftime("%Y-%m-%d") if disp else None,
        deliv.strftime("%Y-%m-%d") if deliv else None,
        st, random.choice(["standard", "standard", "standard", "express"]),
        op, f"TRK{random.randint(10**9, 10**10-1)}" if disp else None,
        REGION[cid]))
    return oid

# ---------- deterministic edge cases (referenced by eval sets) ----------
EDGE = []
india = [x for x in CIDS if REGION[x] == "IN"]
eu    = [x for x in CIDS if REGION[x] == "EU"]

# Boundary: opened electronics, 14-day window
EDGE.append(("day 12 of 14 - opened laptop, ELIGIBLE",
             make_order(india[0], "PB14", 12, opened=1, payment="credit_card",
                        force_id="PAC-2026-12345")))
EDGE.append(("day 14 of 14 - opened laptop, ELIGIBLE (last day)",
             make_order(india[1], "PB14", 14, opened=1, payment="upi",
                        force_id="PAC-2026-12346")))
EDGE.append(("day 15 - opened laptop, EXPIRED by one day",
             make_order(india[2], "PB16", 15, opened=1, payment="upi",
                        force_id="PAC-2026-12347")))
EDGE.append(("day 22 - SEALED laptop, still eligible on 30-day window",
             make_order(india[3], "PB14L", 22, opened=0, payment="net_banking",
                        force_id="PAC-2026-12348")))
EDGE.append(("day 31 - sealed laptop, EXPIRED",
             make_order(india[4], "PB14", 31, opened=0, payment="upi",
                        force_id="PAC-2026-12349")))
# DOA window
EDGE.append(("delivered 1 day ago - inside 48h DOA window",
             make_order(india[5], "PV27", 1, opened=1, payment="upi",
                        force_id="PAC-2026-12350")))
EDGE.append(("day 8 - the 48h-to-14d GREY ZONE (DEFECT-12)",
             make_order(india[6], "PB14", 8, opened=1, payment="credit_card",
                        force_id="PAC-2026-12351")))
# EMI cases
EDGE.append(("no-cost EMI, day 10 - DEFECT-07 refund arithmetic",
             make_order(india[7], "PB14", 10, opened=1, payment="emi",
                        no_cost_emi=1, force_id="PAC-2026-12352")))
EDGE.append(("standard EMI, day 6 - DEFECT-06 interest not refunded",
             make_order(india[8], "PB16", 6, opened=1, payment="emi",
                        force_id="PAC-2026-12353")))
# EU override
EDGE.append(("EU customer, opened, day 10 - DEFECT-03 override applies",
             make_order(eu[0], "PB14", 10, opened=1, payment="credit_card",
                        force_id="PAC-2026-12354")))
EDGE.append(("EU customer, opened, day 16 - past EU 14d window",
             make_order(eu[1], "PPX", 16, opened=1, payment="credit_card",
                        force_id="PAC-2026-12355")))
# Warranty
EDGE.append(("Pacify laptop, 8 months old - warranty, Pacify-administered",
             make_order(india[9], "PB14", 240, opened=1, payment="upi",
                        force_id="PAC-2026-12356")))
EDGE.append(("THIRD-PARTY laptop, 8 months old - DEFECT-09 manufacturer route",
             make_order(india[10], "NWU15", 240, opened=1, payment="credit_card",
                        force_id="PAC-2026-12357")))
EDGE.append(("Pacify phone, 25 months old - warranty EXPIRED by 1 month",
             make_order(india[11], "PPX", 760, opened=1, payment="upi",
                        force_id="PAC-2026-12358")))
EDGE.append(("accessory, 7 months old - 6-month warranty EXPIRED",
             make_order(india[12], "PSP", 210, opened=1, payment="upi",
                        force_id="PAC-2026-12359")))
EDGE.append(("refurbished laptop, 5 months - 6-month warranty ACTIVE",
             make_order(india[13], "PB14R", 150, opened=1, payment="upi",
                        force_id="PAC-2026-12360")))
# Awkward states
EDGE.append(("cancelled before dispatch",
             make_order(india[14], "PV27", None, status="cancelled",
                        force_id="PAC-2026-12361")))
EDGE.append(("in transit - no delivery date yet",
             make_order(india[15], "PT11", None, status="in_transit",
                        force_id="PAC-2026-12362")))
EDGE.append(("delivery FAILED after 3 attempts",
             make_order(india[16], "PPX", None, status="delivery_failed",
                        force_id="PAC-2026-12363")))
EDGE.append(("refund in progress - card, initiated 3 days ago",
             make_order(india[17], "PB14L", 25, status="refund_in_progress",
                        opened=1, payment="credit_card", force_id="PAC-2026-12364")))
EDGE.append(("already returned and refunded",
             make_order(india[18], "PT11", 40, status="returned", opened=1,
                        payment="upi", force_id="PAC-2026-12365")))
EDGE.append(("COD order - refund needs bank details",
             make_order(india[19], "PV27", 9, opened=1, payment="cod",
                        force_id="PAC-2026-12366")))
EDGE.append(("BULK order 6 units - 7-day window, day 5 ELIGIBLE",
             make_order(india[20], "PKL", 5, opened=1, payment="net_banking",
                        qty=6, force_id="PAC-2026-12367")))
EDGE.append(("BULK order 8 units - day 9, EXPIRED (7-day window)",
             make_order(india[21], "PT11", 9, opened=1, payment="credit_card",
                        qty=8, force_id="PAC-2026-12368")))
EDGE.append(("accessory day 25 - 30d window, no restocking fee",
             make_order(india[22], "PKL", 25, opened=1, payment="upi",
                        force_id="PAC-2026-12369")))
EDGE.append(("stuck in transit 9 days - trace required",
             make_order(india[23], "PB16", None, status="in_transit",
                        force_id="PAC-2026-12370")))

# ---------- bulk random orders ----------
for _ in range(1975):
    cid = random.choice(CIDS)
    sku = random.choice(SKUS)
    st = random.choice(STATUSES)
    if st in ("delivered", "returned", "refund_in_progress"):
        make_order(cid, sku, random.randint(1, 400), status=st)
    else:
        make_order(cid, sku, None, status=st)

c.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", orders)

# update customer order counts
c.execute("""UPDATE customers SET total_orders =
             (SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.customer_id)""")

# ---------- returns ----------
rets = []
c.execute("SELECT order_id, sku, unit_price, quantity, is_opened, status, delivery_date, region FROM orders WHERE status IN ('returned','refund_in_progress')")
for n, (oid, sku, up, qty, opened, st, dd, reg) in enumerate(c.fetchall()):
    reason = random.choice(["change_of_mind", "defective", "wrong_item", "damaged_in_transit"])
    fee_pct = 0.10 if (opened and reason == "change_of_mind" and CAT[sku] != "accessory") else 0.0
    if reg == "EU":
        fee_pct = 0.0
    fee = round(up * qty * fee_pct, 2)
    rship = 450.0 if reason == "change_of_mind" else 0.0
    amt = round(up * qty - fee - rship, 2)
    req = (datetime.strptime(dd, "%Y-%m-%d") + timedelta(days=random.randint(1, 12))).strftime("%Y-%m-%d")
    rets.append((f"RET-{20000+n}", oid, req, reason,
                 "completed" if st == "returned" else "refund_initiated",
                 fee, rship, amt,
                 "credited" if st == "returned" else "initiated",
                 req))
c.executemany("INSERT INTO returns VALUES (?,?,?,?,?,?,?,?,?,?)", rets)

# ---------- warranty claims ----------
CODES = ["BAT-119", "DSP-014", "THRM-12", "KEY-018", "STO-440", "MEM-221",
         "WIFI-503", "ERR-DP-0x004", None, None]
FAULTS = ["will not power on", "battery drains within an hour",
          "screen flickers under load", "backlight failed",
          "keyboard unresponsive", "random shutdowns", "wifi drops constantly",
          "monitor goes black intermittently", "fan making grinding noise",
          "dead pixels visible on white background"]
wcs = []
c.execute("SELECT order_id, delivery_date FROM orders WHERE status='delivered' AND delivery_date IS NOT NULL LIMIT 200")
rows = c.fetchall()
for n, (oid, dd) in enumerate(random.sample(rows, 120)):
    raised = (datetime.strptime(dd, "%Y-%m-%d") + timedelta(days=random.randint(50, 300)))
    if raised > TODAY:
        raised = TODAY - timedelta(days=random.randint(1, 30))
    wcs.append((f"WCL-{30000+n}", oid, raised.strftime("%Y-%m-%d"),
                random.choice(FAULTS), random.choice(CODES),
                random.choice(["open", "in_service", "resolved", "resolved", "rejected"]),
                random.choice(["repaired", "replaced", "refunded", "not_covered", None])))
c.executemany("INSERT INTO warranty_claims VALUES (?,?,?,?,?,?,?)", wcs)

con.commit()

print("=" * 58)
print("pacify.db built")
print("=" * 58)
for t in ["customers", "products", "orders", "returns", "warranty_claims"]:
    n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:20s} {n:6d}")
print("\nOrder status distribution:")
for s, n in c.execute("SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY 2 DESC"):
    print(f"  {s:22s} {n:5d}")
print("\nDeterministic edge cases (stable IDs for eval sets):")
for desc, oid in EDGE:
    print(f"  {oid}  {desc}")
con.close()
