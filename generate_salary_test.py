import pandas as pd

rows = [
    {'Date': '01/03/2024', 'Narration': 'SALARY CREDIT - MARCH 2024', 'Chq./Ref.No.': '', 'Value Date': '01/03/2024', 'Withdrawal Amt.': '', 'Deposit Amt.': 80000, 'Closing Balance': 180000},
    {'Date': '05/03/2024', 'Narration': 'EMI PAYMENT HDFC LOAN', 'Chq./Ref.No.': '', 'Value Date': '05/03/2024', 'Withdrawal Amt.': 25000, 'Deposit Amt.': '', 'Closing Balance': 155000},
    {'Date': '01/03/2024', 'Narration': 'NEFT/Landlord Rent March', 'Chq./Ref.No.': '', 'Value Date': '01/03/2024', 'Withdrawal Amt.': 20000, 'Deposit Amt.': '', 'Closing Balance': 135000},
    {'Date': '10/03/2024', 'Narration': 'UPI/Swiggy Order', 'Chq./Ref.No.': '', 'Value Date': '10/03/2024', 'Withdrawal Amt.': 850, 'Deposit Amt.': '', 'Closing Balance': 134150},
    {'Date': '12/03/2024', 'Narration': 'ATM Withdrawal', 'Chq./Ref.No.': '', 'Value Date': '12/03/2024', 'Withdrawal Amt.': 10000, 'Deposit Amt.': '', 'Closing Balance': 124150},
    {'Date': '17/03/2024', 'Narration': 'NEFT/Unknown Vendor Pvt Ltd', 'Chq./Ref.No.': '', 'Value Date': '17/03/2024', 'Withdrawal Amt.': 99000, 'Deposit Amt.': '', 'Closing Balance': 25150},
    {'Date': '20/03/2024', 'Narration': 'NEFT/Unknown Vendor Pvt Ltd', 'Chq./Ref.No.': '', 'Value Date': '20/03/2024', 'Withdrawal Amt.': 98500, 'Deposit Amt.': '', 'Closing Balance': 126650},
    {'Date': '23/03/2024', 'Narration': 'NEFT/Unknown Vendor Pvt Ltd', 'Chq./Ref.No.': '', 'Value Date': '23/03/2024', 'Withdrawal Amt.': 97000, 'Deposit Amt.': '', 'Closing Balance': 29650},
]

df = pd.DataFrame(rows)
df.to_csv('data/sample/test_salary.csv', index=False)
print('Generated salary test file')
print('Configure: salary=80000, EMI=25000')
print('Expected: salary/EMI NOT flagged, Unknown Vendor split billing flagged')