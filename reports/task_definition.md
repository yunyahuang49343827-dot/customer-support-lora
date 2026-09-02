# Stage C2 Task Definition

## 1. Evidence Basis

This contract is derived from Stage C1 artifacts for `bitext/Bitext-customer-support-llm-chatbot-training-dataset`: 26,872 loaded rows, 27 canonical intents, 11 canonical categories, balanced intent counts, zero detected normalized-instruction label conflicts, substantial normalized/template duplicate groups, and relatively long source responses. No external 10-category claim is used.

## 2. Input Contract

The model receives exactly one **customer support message**: the customer-authored `instruction` string. It must be non-empty after trimming and is treated as untrusted customer text.

At inference, the model must not receive `flags`, ground-truth `intent`, ground-truth `category`, or the reference `response`. Dataset placeholders such as `{{Order Number}}` may remain part of the customer message; they are text, not trusted metadata or permission to invent values.

## 3. Output Contract

The complete output must be exactly one valid JSON object with exactly these keys:

```json
{
  "intent": "<allowed canonical intent>",
  "category": "<allowed canonical category>",
  "needs_human": false,
  "response": "<non-empty customer-facing response>"
}
```

Markdown fences, prefixes, suffixes, comments, multiple JSON values, missing keys, and extra keys are forbidden. `intent` and `category` must use the canonical vocabularies exactly, `needs_human` must be a JSON boolean, and `response` must contain at least one non-whitespace character. The category must also equal the deterministic category mapped from the selected intent.

The machine-readable contract is `configs/output_schema.json` using JSON Schema Draft 2020-12.

## 4. Intent and Category Taxonomies

- Canonical intents: 27; defined in `configs/intent_taxonomy.json`.
- Canonical categories: 11; defined in `configs/category_taxonomy.json`.
- Canonical category names: `ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, `SUBSCRIPTION`.
- Labels are copied from Stage C1 artifacts without renaming or merging.

## 5. Deterministic Intent → Category Mapping

Every observed intent maps to exactly one category; there are no ambiguous or many-to-many intent mappings in Stage C1.

| Intent | Category | Dataset rows |
|---|---|---:|
| `cancel_order` | `ORDER` | 998 |
| `change_order` | `ORDER` | 997 |
| `change_shipping_address` | `SHIPPING` | 973 |
| `check_cancellation_fee` | `CANCEL` | 950 |
| `check_invoice` | `INVOICE` | 1,000 |
| `check_payment_methods` | `PAYMENT` | 999 |
| `check_refund_policy` | `REFUND` | 997 |
| `complaint` | `FEEDBACK` | 1,000 |
| `contact_customer_service` | `CONTACT` | 1,000 |
| `contact_human_agent` | `CONTACT` | 999 |
| `create_account` | `ACCOUNT` | 997 |
| `delete_account` | `ACCOUNT` | 995 |
| `delivery_options` | `DELIVERY` | 995 |
| `delivery_period` | `DELIVERY` | 999 |
| `edit_account` | `ACCOUNT` | 1,000 |
| `get_invoice` | `INVOICE` | 999 |
| `get_refund` | `REFUND` | 997 |
| `newsletter_subscription` | `SUBSCRIPTION` | 999 |
| `payment_issue` | `PAYMENT` | 999 |
| `place_order` | `ORDER` | 998 |
| `recover_password` | `ACCOUNT` | 995 |
| `registration_problems` | `ACCOUNT` | 999 |
| `review` | `FEEDBACK` | 997 |
| `set_up_shipping_address` | `SHIPPING` | 997 |
| `switch_account` | `ACCOUNT` | 1,000 |
| `track_order` | `ORDER` | 995 |
| `track_refund` | `REFUND` | 998 |

## 6. Escalation Contract

`needs_human` ground truth does not exist in the Bitext dataset. Evaluation therefore uses the explicit intent-level policy in `configs/escalation_policy.json`, never a label inferred from reference response wording.

The formally confirmed policy contains 6 escalated intents and 21 non-escalated intents. `needs_human = true` applies exactly to `complaint`, `contact_customer_service`, `contact_human_agent`, `delete_account`, `get_refund`, and `payment_issue`; every other canonical intent is `false`.

Refund-related distinctions are explicit: `check_refund_policy = false`, `track_refund = false`, and `get_refund = true`. Password recovery uses secure self-service guidance, so `recover_password = false`. Likewise, routine cancellation, order-change, and shipping-address-change intents remain `false`; responses may explain authorized workflows but must never claim an action was completed.

## 7. Response Constraints

A compliant response must:

- directly address the customer intent and remain concise;
- provide useful next-step guidance where possible;
- avoid inventing company-specific policies, guarantees, eligibility decisions, fees, timelines, contact details, or capabilities;
- never claim that a refund, cancellation, payment, order, shipment, or account action was completed when it was not actually executed;
- never invent reference, case, transaction, order, tracking, invoice, or refund numbers;
- contain no metadata or schema fields outside the enclosing four-key JSON object;
- when `needs_human` is `true`, clearly and calmly explain why authorized or human assistance is needed and give a safe handoff-oriented next step;
- when `needs_human` is `false`, avoid unnecessary forced escalation and provide self-service or informational guidance;
- avoid requesting secrets such as passwords, full payment-card data, or authentication codes.

Stage C1 found a mean reference-response length of 634.104 characters (104.789 whitespace-delimited words). Copying or closely matching those long responses is **not** a success criterion; relevance, safety, honesty, and concision are.

## 8. Stage Boundary

Stage C2 defines specifications, configuration, and validation logic only. It creates no Train, Validation, Dev, or Locked Test split and performs no model loading, inference, training, prompt tuning, or evaluation run.
