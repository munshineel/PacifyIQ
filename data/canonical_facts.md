# Pacify — Canonical Facts

**This file is the single source of truth for the entire PacifyIQ knowledge base.**

Every document, every database row, and every evaluation answer must be consistent
with this table. Any deviation found in a document is either:

1. A **planted defect** — logged in `PLANTED_DEFECTS.md` with an ID, or
2. A **bug** — fix it.

If it is not in `PLANTED_DEFECTS.md`, it is a bug. There is no third category.

Company: **Pacify Electronics Pvt. Ltd.**
Domain: `www.pacify.com` · Support: `support@pacify.com`
Order ID format: `PAC-2026-NNNNN` · Ticket ID format: `TKT-NNNNN`
Currency: INR (₹) · Primary market: India · Secondary: EU (12 countries)

---

## 1. Returns

| Fact | Canonical value |
|---|---|
| Return window — sealed / unopened | 30 calendar days from delivery |
| Return window — **opened electronics** | **14 calendar days from delivery** |
| Return window — accessories (opened or not) | 30 calendar days from delivery |
| Return window — bulk / business orders (>= 5 units) | 7 calendar days from delivery |
| Window start point | Date of **delivery** (not dispatch, not purchase) |
| Window ending on Sunday / public holiday | Extends to next business day |
| Non-returnable | Software licence keys, redeemed gift cards, opened SIM/eSIM, custom-configured units |
| Damage / DOA reporting window | **48 hours from delivery** |
| Restocking fee — opened laptop / monitor | 10% of item value |
| Restocking fee — opened phone / tablet | 10% of item value |
| Restocking fee — accessories | Nil |
| Restocking fee waived when | Defective, damaged in transit, wrong item shipped |
| Return shipping — change of mind | Rs 450, deducted from refund |
| Return shipping — defect / wrong item | Free, Pacify-arranged pickup |
| Inspection period after receipt at warehouse | 2 business days |
| Definition of "opened" | Outer retail seal broken (see DEFECT-04) |

## 2. Refunds

| Fact | Canonical value |
|---|---|
| Refund — UPI | 3-5 business days |
| Refund — credit / debit card | 5-7 business days |
| Refund — net banking | 5-7 business days |
| Refund — EMI | 7-14 business days (bank dependent) |
| Refund — COD (to bank transfer) | 7-10 business days |
| Store credit alternative | Instant, +5% bonus value |
| Original outbound shipping refunded? | Only if defective / wrong item / DOA |
| EMI interest already accrued | **Not refunded** (see DEFECT-06) |
| No-cost EMI on return | Refund is of the **discounted** amount (see DEFECT-07) |
| Price-drop protection | 7 days from delivery, difference refunded |
| Refund clock starts | On completion of warehouse inspection |

### Refund fee waterfall (canonical formula)

```
refund = item_price
       - restocking_fee        (10% if opened & change-of-mind, else 0)
       - return_shipping       (450 if change-of-mind, else 0)
       + original_shipping     (only if defective / wrong item / DOA)
```

## 3. Warranty

| Fact | Canonical value |
|---|---|
| Warranty — standard (third-party brands) | 12 months from delivery |
| Warranty — Pacify-branded hardware | 24 months from delivery |
| Warranty — accessories | 6 months from delivery |
| Warranty — refurbished units | 6 months from delivery |
| PacifyCare+ extended | +12 or +24 months; purchasable within 30 days of delivery |
| Warranty start point | Date of **delivery** |
| Battery coverage | Health below 80% within 12 months |
| **Dead / stuck pixel threshold** | **5 or more** = replacement (see DEFECT-05) |
| Warranty exclusions | Liquid damage, physical damage, unauthorised repair, cosmetic wear, software / OS issues, consumables |
| Repair turnaround | 7-14 business days |
| Responsibility — Pacify-branded | Pacify handles end to end |
| Responsibility — third-party brands | Manufacturer service centre; Pacify facilitates only |
| Replacement out of stock | Refund at original price paid |
| Warranty transferable on resale | Yes, with original invoice |

## 4. Shipping

| Fact | Canonical value |
|---|---|
| Order processing time (before dispatch) | 24 hours (1 business day) |
| Shipping — standard | 3-7 business days |
| Shipping — express | 1-2 business days |
| Free shipping threshold | Orders >= Rs 5,000 |
| Standard shipping cost below threshold | Rs 99 |
| Express shipping cost | Rs 299 |
| Address change cutoff | Before dispatch only |
| Failed delivery attempts before return-to-origin | 3 |
| Delay compensation threshold | 5 business days beyond committed date |
| Serviceable — India | All serviceable pincodes |
| Serviceable — EU | 12 countries (see shipping_policy S1) |
| White-glove delivery | Monitors 32 inch and above |

## 5. Payments

| Fact | Canonical value |
|---|---|
| EMI minimum order value | Rs 15,000 |
| EMI tenures | 3, 6, 9, 12, 18, 24 months |
| COD maximum order value | Rs 50,000 |
| COD handling fee | Rs 50 |
| Failed payment auto-reversal | 5-7 business days (see DEFECT-08) |
| Double charge resolution | 7-10 business days after verification |
| Accepted methods | UPI, credit card, debit card, net banking, EMI, COD, gift card, store credit |
| GST invoice | Issued on dispatch, emailed within 24 hours |

## 6. Customer service

| Fact | Canonical value |
|---|---|
| Support hours | 09:00-21:00 IST, Monday to Saturday |
| First response SLA — chat | 5 minutes |
| First response SLA — email | 24 hours |
| First response SLA — phone | Immediate during hours |
| Complaint resolution SLA | 7 business days |
| Grievance Officer response | 30 days (statutory) |
| Grievance Officer | Ms. R. Iyer, grievance@pacify.com |
| Escalation tiers | L1 agent -> L2 specialist -> Grievance Officer |
| Identity verification for account actions | Registered-email OTP **plus** last 4 digits of payment method **plus** one order ID |
| Data deletion request completion | 30 days |
| AI assistant disclosure | Disclosed on request; human agent available on request at any time |

## 7. EU regional (overrides sections 1-2 for EU customers)

| Fact | Canonical value |
|---|---|
| Statutory withdrawal period | 14 calendar days from delivery |
| Reason required | No |
| Applies to opened items | **Yes** — overrides the "opened" distinction |
| Return shipping | Customer pays unless defective |
| Refund deadline after receipt | 14 calendar days |
| Restocking fee permitted | No |

## 8. Product lineup (fictional SKUs — no real-world equivalents)

| SKU | Category | Price (Rs) | Brand |
|---|---|---|---|
| Pacify ProBook 14 | Laptop | 64,900 | Pacify |
| Pacify ProBook 16 | Laptop | 89,900 | Pacify |
| Pacify ProBook 14 Lite | Laptop | 47,900 | Pacify |
| Pacify Phone X | Phone | 38,900 | Pacify |
| Pacify Phone X Pro | Phone | 54,900 | Pacify |
| Pacify Tab 11 | Tablet | 29,900 | Pacify |
| Pacify Vision 27 | Monitor | 24,900 | Pacify |
| Pacify Vision 32 | Monitor | 41,900 | Pacify |
| Pacify KeyLite | Accessory | 3,499 | Pacify |
| Pacify SoundPods | Accessory | 5,999 | Pacify |
| Northwind Ultra 15 | Laptop | 72,900 | Northwind (third party) |
| Kestrel Note 9 | Phone | 31,900 | Kestrel (third party) |

Third-party brands exist specifically to exercise the warranty responsibility
split (canonical section 3) — see DEFECT-09.

## 9. Error codes (canonical reference)

| Code | Meaning | Surface it appears on | User-fixable |
|---|---|---|---|
| PAY-402 | Payment gateway timeout | Checkout screen | Yes |
| PAY-511 | 3-D Secure authentication failure | Bank redirect page | Yes |
| PAY-207 | Insufficient funds | Checkout screen | Yes |
| ERR-DP-0x004 | DisplayPort handshake failure | Monitor on-screen display | Yes |
| ERR-DP-0x011 | Unsupported refresh rate | Monitor on-screen display | Yes |
| BAT-119 | Battery health critical (below 60%) | System tray notification | No - warranty |
| BAT-042 | Charger not recognised | System tray notification | Yes |
| WIFI-503 | Wireless driver initialisation failure | OS network dialog | Yes |
| SYS-0x0000007B | Boot device inaccessible | Boot / stop screen | Sometimes |
| THRM-88 | Sustained thermal throttling | Pacify Diagnostics app | Yes |
| DSP-014 | Panel backlight failure | Visible symptom, no code shown | No - warranty |
| AUD-330 | Audio driver conflict | OS sound dialog | Yes |

## 10. Facts deliberately ABSENT from the corpus

These are the basis of the unanswerable / abstention evaluation set. The corpus
must genuinely not contain them. If any document mentions them, the abstention
test is broken.

- Student or education discounts
- Trade-in or buy-back programme
- Physical retail store locations
- B2B leasing or rental
- Loyalty / points programme
- Gift wrapping service
- Carbon-neutral or sustainability commitments
- Insurance products (accidental damage cover beyond PacifyCare+)
