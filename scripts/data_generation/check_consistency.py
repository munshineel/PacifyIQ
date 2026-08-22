"""Verify corpus matches canonical_facts.md. Planted defects are expected to appear."""
import os, sys
from pathlib import Path
# project-relative: <root>/data
DATA_ROOT = str(Path(__file__).resolve().parents[2] / "data")

import glob, re
S=f"{DATA_ROOT}/_source"
docs={}
for p in glob.glob(f"{S}/*.md"):
    docs[p.split("/")[-1].replace(".md","")]=open(p,encoding="utf-8").read()

CHECKS=[
 ("14 calendar days opened","return_policy_v2",r"14 calendar days",True),
 ("30 calendar days sealed","return_policy_v2",r"30 calendar days",True),
 ("48 hours DOA","return_policy_v2",r"48 hours",True),
 ("10% restocking","return_policy_v2",r"10% of item value",True),
 ("Rs 450 return shipping","return_policy_v2",r"Rs 450",True),
 ("5-7 card refund","refund_policy",r"5-7 business days",True),
 ("3-5 UPI refund","refund_policy",r"3-5 business days",True),
 ("EMI interest not refunded","refund_policy",r"not refunded",True),
 ("5 dead pixels","warranty_policy",r"\*\*5 or more\*\*",True),
 ("24 months Pacify","warranty_policy",r"24 months",True),
 ("12 months third party","warranty_policy",r"12 months",True),
 ("free ship 5000","shipping_policy",r"Rs 5,000",True),
 ("EMI min 15000","payment_policy",r"Rs 15,000",True),
 ("COD max 50000","payment_policy",r"Rs 50,000",True),
 ("verification 3 limbs","customer_service_policy",r"all three",True),
 ("EU 14 days","eu_regional_addendum",r"14 calendar days",True),
]
print("CONSISTENCY vs canonical_facts.md")
print("-"*60)
bad=0
for name,doc,pat,want in CHECKS:
    found=bool(re.search(pat,docs[doc],re.I))
    ok="PASS" if found==want else "FAIL"
    if ok=="FAIL": bad+=1
    print(f"  {ok}  {name:28s} in {doc}")

print("\nPLANTED DEFECTS present?")
print("-"*60)
D=[("DEFECT-01 30-day guarantee","shipping_policy",r"30-day satisfaction guarantee"),
   ("DEFECT-02 v1 says 30 days","return_policy_v1_ARCHIVED",r"within 30 days"),
   ("DEFECT-03 EU override","eu_regional_addendum",r"whether or not the item has been opened"),
   ("DEFECT-05 pixel threshold","warranty_policy",r"fewer than 5"),
   ("DEFECT-06 EMI interest","refund_policy",r"Interest, processing charges"),
   ("DEFECT-07 no-cost EMI","refund_policy",r"discounted amount actually paid"),
   ("DEFECT-08 5-7 collision","payment_policy",r"5 to 7 business days"),
   ("DEFECT-09 brand split","warranty_policy",r"manufacturer's authorised service network"),
   ("DEFECT-12 grey zone","warranty_policy",r"at the customer's election"),
   ("DEFECT-18 embedded injection","product_faq",r"automated indexing systems")]
for name,doc,pat in D:
    found=bool(re.search(pat,docs[doc],re.I))
    print(f"  {'PRESENT' if found else 'MISSING':8s} {name}")
    if not found: bad+=1

print("\nDELIBERATE OMISSIONS absent from corpus?")
print("-"*60)
allt=" ".join(docs.values()).lower()
for term in ["student discount","trade-in","trade in program","physical store","leasing","loyalty point","gift wrap","carbon neutral"]:
    hit=term in allt
    print(f"  {'LEAK!' if hit else 'clean':8s} {term}")
    if hit: bad+=1
print(f"\n{'ALL CHECKS PASSED' if bad==0 else f'{bad} ISSUES'}")
