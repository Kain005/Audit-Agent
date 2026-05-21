from api.report_generator import ReportGenerator

report = {
    'risk_score': 46,
    'summary': 'Risk score 46/100 based on 2 high severity findings.',
    'documents_processed': 1,
    'total_transactions': 9,
    'anomalies': [
        {
            'severity': 'HIGH',
            'finding_type': 'split_billing',
            'human_readable_reason': 'Vendor received 3 payments totaling Rs.290000',
            'amount_involved': 290000
        }
    ],
    'policy_violations': []
}

pdf = ReportGenerator().generate_pdf(report)
with open('test_report_xhtml2pdf.pdf', 'wb') as f:
    f.write(pdf)
print('Wrote test_report_xhtml2pdf.pdf, bytes:', len(pdf))
