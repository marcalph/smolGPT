
# take home data pipeline

## context

A customer subscribes at a fixed annual price and is committed for 12 months.
During the commitment period, the customer will pay the amount specified in their contract.
In the 13th month, the customer's contract will be renewed, but with a price increase because the contract’s age has changed.
Every customer has a client portal (login/password) where they can submit reimbursement requests.

## user stories

As a customer, I would like to:

- Be notified one month before the renewal.
- Have the option to accept or decline the renewal directly from my client portal.
- Avoid being charged twice at the time of renewal.
- Have the possibility to renegotiate the options of my contract, which will trigger a new price calculation.

## Renewal constraints

It is not possible to create a subscription for a future date; that is, the new contract must be activated at the moment the old contract expires.
