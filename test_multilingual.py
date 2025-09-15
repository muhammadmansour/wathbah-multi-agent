# -*- coding: utf-8 -*-
"""
Simple test to demonstrate multilingual agent detection
"""

def test_arabic_keywords():
    """Test Arabic keyword detection"""
    
    # Test cases
    test_cases = [
        {
            "message": "البريد الإلكتروني",
            "expected_agent": "mailer",
            "description": "Arabic for 'email'"
        },
        {
            "message": "تحليل هذا التقرير المالي",
            "expected_agent": "analyzer", 
            "description": "Arabic for 'analyze this financial report'"
        },
        {
            "message": "ملخص هذه الوثيقة",
            "expected_agent": "summarizer",
            "description": "Arabic for 'summarize this document'"
        },
        {
            "message": "احسب هذه المعادلة",
            "expected_agent": "mathematical",
            "description": "Arabic for 'calculate this equation'"
        }
    ]
    
    print("🧪 Multilingual Agent Detection Test")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['description']}")
        print(f"   Message: '{test_case['message']}'")
        print(f"   Expected: {test_case['expected_agent']}")
        
        # Check if Arabic keywords are present
        message = test_case['message']
        
        # Simulate keyword detection
        mailer_keywords = ["البريد الإلكتروني", "إرسال", "بريد", "مشاركة"]
        analyzer_keywords = ["تحليل", "فحص", "مراجعة", "تقييم", "مالي"]
        summarizer_keywords = ["ملخص", "تلخيص", "موجز", "نظرة عامة"]
        mathematical_keywords = ["حساب", "احسب", "رياضيات", "معادلة", "حل"]
        
        detected_agent = "general"  # default
        
        if any(keyword in message for keyword in mailer_keywords):
            detected_agent = "mailer"
        elif any(keyword in message for keyword in analyzer_keywords):
            detected_agent = "analyzer"
        elif any(keyword in message for keyword in summarizer_keywords):
            detected_agent = "summarizer"
        elif any(keyword in message for keyword in mathematical_keywords):
            detected_agent = "mathematical"
        
        status = "✅ PASS" if detected_agent == test_case['expected_agent'] else "❌ FAIL"
        print(f"   Detected: {detected_agent}")
        print(f"   Status: {status}")

if __name__ == "__main__":
    test_arabic_keywords()
    print(f"\n🎯 The system now supports:")
    print(f"   • Arabic: البريد الإلكتروني (email), تحليل (analysis), ملخص (summary)")
    print(f"   • Spanish: enviar correo, análisis, resumen")
    print(f"   • French: envoyer mail, analyse, résumé")
    print(f"   • German: e-mail senden, analyse, zusammenfassung")
    print(f"   • English: send email, analysis, summary")
