# PACIFY ELECTRONICS PRIVATE LIMITED
## Payments Policy

**Document reference:** POL-PAY-001
**Version:** 1.3
**Effective from:** 15 January 2026
**Owner:** Finance Operations

---

## S1. Accepted payment methods

S1.1 Pacify accepts the following methods:

(a) Unified Payments Interface (UPI);
(b) credit cards issued by Visa, Mastercard, RuPay, and American Express;
(c) debit cards issued by Visa, Mastercard, and RuPay;
(d) net banking with 48 participating banks;
(e) equated monthly instalments, subject to S2;
(f) cash on delivery, subject to S3;
(g) Pacify gift cards;
(h) Pacify store credit.

S1.2 Payment is taken in Indian Rupees for Indian deliveries and in Euro for deliveries to the European Union.

S1.3 Pacify does not accept cheques, demand drafts, cryptocurrency, or payment by international wire transfer.

S1.4 A single order may be settled using at most two methods, of which at most one may be a card.

---

## S2. Equated monthly instalments

S2.1 EMI is available on orders of **Rs 15,000 and above**.

S2.2 Available tenures are 3, 6, 9, 12, 18, and 24 months. Availability of a given tenure depends on the issuing bank.

S2.3 EMI is available on credit cards from participating banks and, for certain banks, on debit cards subject to the bank's own eligibility assessment.

S2.4 Interest is set by the issuing bank and not by Pacify. The rate applicable is displayed at checkout before confirmation.

S2.5 Where an offer is described as "no-cost EMI", the interest chargeable by the bank is absorbed by Pacify and applied as a discount to the item price at the point of sale. The customer is charged instalments totalling the original price, of which a portion represents bank interest funded by that discount.

S2.6 The consequence of S2.5 on a return is dealt with at POL-REF-001 S7.3. In summary, the refund is of the discounted amount actually paid, and the absorbed interest is not returned to the customer.

S2.7 A processing fee may be levied by the issuing bank on EMI conversion. Such fee is not refundable by Pacify.

---

## S3. Cash on delivery

S3.1 Cash on delivery is available on orders up to **Rs 50,000** in value.

S3.2 A handling fee of **Rs 50** applies to every cash on delivery order. The fee is not refundable in any circumstances, including where the order is returned.

S3.3 Cash on delivery is not available at restricted pincodes identified under POL-SHP-001 S1.4, nor for made-to-order items, nor for orders including a gift card.

S3.4 The carrier accepts cash and, at most delivery locations, card and UPI payment on the doorstep.

S3.5 Refund of a cash on delivery order requires bank details from the customer. See POL-REF-001 S8.

---

## S4. Failed payments and automatic reversal

S4.1 A payment may fail for reasons including insufficient funds, an expired card, a declined authentication, a bank-side timeout, or a gateway timeout.

S4.2 Where a payment fails, no order is created. The customer may retry immediately using the same or a different method.

S4.3 In some failure modes the customer's bank places a hold on the funds, or debits and then must reverse the amount. Where this occurs, the amount is **automatically reversed by the bank within 5 to 7 business days**. No action by the customer or by Pacify is required, and Pacify cannot accelerate the reversal because Pacify never received the funds.

S4.4 The reversal at S4.3 is not a refund. A refund arises only where an order was completed and subsequently returned, and is governed by POL-REF-001. Customers frequently conflate the two. The distinguishing question is whether an order reference exists: if no order was created, the matter is a reversal under this section.

S4.5 Where an amount has not been reversed after 7 business days, the customer should raise the matter with their bank quoting the transaction reference, and notify Pacify so that the gateway record can be supplied in support.

---

## S5. Double charges

S5.1 A double charge arises where two settled transactions correspond to a single order.

S5.2 On report, Pacify verifies against gateway records within 2 business days.

S5.3 Where a double charge is confirmed, the duplicate is refunded within **7 to 10 business days** of verification, to the original instrument.

S5.4 Two authorisations of which only one has settled do not constitute a double charge. The unsettled authorisation lapses under S4.3.

---

## S6. Card storage and security

S6.1 Pacify does not store card numbers. Cards saved for future use are tokenised in accordance with Reserve Bank of India requirements, and the token is held by the card network.

S6.2 Pacify support staff never request a full card number, CVV, PIN, or one-time password. A request purporting to come from Pacify for any of these should be treated as fraudulent and reported to support@pacify.com.

S6.3 Saved payment methods may be removed from the Payments section of the customer's Pacify account.

---

## S7. Gift cards and store credit

S7.1 Gift cards are issued in denominations from Rs 500 to Rs 50,000, are valid for 24 months from issue, and may be used across multiple orders until exhausted.

S7.2 A gift card, once redeemed against an order, is not returnable. See POL-RET-002 S5.1(b).

S7.3 Store credit arises from a refund election under POL-REF-001 S6, from delay compensation under POL-SHP-001 S8.2, or from a goodwill adjustment. It is valid for 24 months, is not transferable, and cannot be converted to cash.

S7.4 Where an order is settled partly by gift card or store credit and is subsequently returned, that portion is refunded as store credit and not as cash.

---

## S8. Invoicing and tax

S8.1 A GST invoice is generated on dispatch and emailed to the registered address within 24 hours.

S8.2 A GSTIN may be supplied at checkout for a business purchase. A GSTIN cannot be added to an invoice after dispatch.

S8.3 Invoices are available for download from the Orders section of the customer's Pacify account for 7 years from the date of purchase.

---

## S9. Price accuracy

S9.1 Prices are as displayed at the time of order confirmation.

S9.2 Where a manifest pricing error occurs, Pacify reserves the right to cancel the affected order before dispatch and refund in full. Pacify will not require a customer to pay a difference after confirmation.

S9.3 Price drop protection is dealt with at POL-REF-001 S10.

---

## S10. Payment error codes

S10.1 The following codes may be displayed at checkout or on the bank authentication page.

| Code | Meaning | What the customer should do |
|---|---|---|
| PAY-402 | Payment gateway timeout. The gateway did not respond within the permitted window. | Wait 10 minutes and retry. If an amount was debited, it reverses under S4.3. |
| PAY-511 | Three-domain-secure authentication failed. The one-time password was not entered, was entered incorrectly, or expired. | Retry, ensuring the registered mobile number is reachable. |
| PAY-207 | Insufficient funds or credit limit exceeded. | Use a different method or reduce the order value. |
| PAY-309 | Card not enrolled for online transactions. | Enable online or international use in the bank's application. |
| PAY-118 | Transaction declined by issuing bank without stated reason. | Contact the issuing bank. Pacify has no visibility of the reason. |
| PAY-604 | Daily transaction limit exceeded. | Retry the following day or use a different method. |

S10.2 Codes beginning PAY- relate to payment processing. Codes relating to device faults are listed in the Technical Support FAQ and are not payment matters, notwithstanding that both are described as error codes.

---

## S11. Related documents

- POL-REF-001 — Refunds Policy
- POL-RET-002 — Returns Policy
- POL-SHP-001 — Shipping and Delivery Policy

---

*Pacify Electronics Private Limited. Issued by Finance Operations.*
