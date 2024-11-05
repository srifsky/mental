from flask import Flask, render_template, request, redirect, url_for, session, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

import matplotlib.pyplot as plt
import pandas as pd
import joblib
import random
import os

app = Flask(__name__)
app.secret_key = 'replace_with_a_random_secret_key'

# Load the pre-trained model
stress_model = joblib.load('stress_condition_model.pkl')
overall_model = joblib.load('mental_health_level_model.pkl')
depression_model = joblib.load('depression_rf_model.pkl')


def interpret_mental_health_score(score):
    if score >= 45:
        return "Good - Your mental health seems to be in a good place. Keep up the positive habits, and make time for activities that support your well-being, like exercise, socializing, and personal hobbies."
    elif 35 <= score < 45:
        return "Moderate - Your mental health is relatively stable, but you may experience occasional stress or low mood. Consider incorporating stress management techniques, like mindfulness or deep breathing exercises, and don’t hesitate to reach out to friends, family, or a mental health professional for support if needed."
    else:
        return "Poor - You may be experiencing significant stress or mental health challenges. It’s important to address these feelings by talking to trusted individuals or seeking support from a mental health professional. Remember, you don’t have to face these challenges alone, and there are resources and people who can help."

# Define the story-based questions and options for the Stress assessment
stress_questions = [
    {
        "question": "คุณตื่นขึ้นมาหลังจากการต่อสู้กับศัตรูในความฝัน คุณรู้สึกพร้อมที่จะลุกขึ้นมาผจญภัยต่อในวันนี้แค่ไหน?",
        "options": ["พลังหมดสิ้น", "พอตัวเองยังไหว", "อาจจะไปได้", "พร้อมออกลุย!", "พลังเต็มเปี่ยมพร้อมรับการต่อสู้!"]
    },
    {
        "question": "คุณมีภารกิจมากมาย แต่ศัตรูโผล่มาไม่หยุด คุณรู้สึกว่าความสามารถของคุณเริ่มลดลงแค่ไหน?",
        "options": ["ใจนิ่งสงบเย็นชา", "เริ่มเครียดเล็กน้อย", "ทนได้พอประมาณ", "ค่อนข้างเครียด", "หนักแน่นไปถึงขั้นที่ทนไม่ไหว!"]
    },
    {
        "question": "ในระหว่างวัน คุณเริ่มรู้สึกถึงพลังเวทมนตร์ที่ตึงเครียดขึ้น คุณรู้สึกว่าพลังของคุณยังเหลืออยู่แค่ไหน?",
        "options": ["เบาสบายเหมือนขนนก", "มีแรงแต่พอน้อย", "พลังลดลงปานกลาง", "พลังเหลือน้อย", "หมดพลังจนไม่สามารถขยับได้"]
    },
    {
        "question": "คุณต้องการตั้งสมาธิเพื่อปล่อยพลังจิต แต่จิตใจของคุณกำลังลอยไปไกล คุณสามารถตั้งสมาธิได้แค่ไหน?",
        "options": ["แน่วแน่เหมือนเหล็กกล้า", "มีฟุ้งซ่านเล็กน้อย", "พอมีสมาธิอยู่บ้าง", "สมาธิเริ่มลดลง", "จิตใจล่องลอยไปในห้วงเวลา"]
    },
    {
        "question": "ขณะที่คุณกำลังปฏิบัติภารกิจ มีคนมาขัดจังหวะ และคุณรู้สึกหงุดหงิดมาก คุณรู้สึกหงุดหงิดแค่ไหน?",
        "options": ["ไม่รู้สึกอะไรเลย", "รู้สึกนิดหน่อย", "หงุดหงิดนิดหน่อย", "หงุดหงิดพอประมาณ", "หงุดหงิดมากจนแทบระเบิด!"]
    },
    {
        "question": "คุณได้รับเชิญให้เข้าร่วมงานเลี้ยงของผู้กล้า แต่คุณไม่แน่ใจว่าจะไปดีหรือไม่ คุณรู้สึกอยากเลี่ยงแค่ไหน?",
        "options": ["อยากไปร่วมฉลอง", "อาจจะไป หรือไม่ไป", "เริ่มอยากหลีกเลี่ยง", "ค่อนข้างอยากหลีกเลี่ยง", "เลี่ยงแน่นอนไม่ไปแน่นอน!"]
    },
    {
        "question": "ถึงเวลาบ่ายแล้ว ความเหนื่อยล้าของการต่อสู้สะสมเพิ่มขึ้น คุณรู้สึกเหนื่อยแค่ไหนในตอนนี้?",
        "options": ["พลังเต็มเปี่ยม", "เริ่มง่วงนิด ๆ", "เริ่มหมดแรง", "เหนื่อยล้าพอสมควร", "พร้อมจะพักผ่อนทันที!"]
    },
    {
        "question": "อาหารมื้อกลางวันดูไม่ช่วยเสริมพลังของคุณเลย คุณรู้สึกว่าความอยากอาหารลดลงไหม?",
        "options": ["ไม่เปลี่ยนแปลง", "ความอยากลดลงเล็กน้อย", "เริ่มเปลี่ยนแปลงบ้าง", "รู้สึกอยากอาหารลดลงมาก", "หิวเหมือนหมาป่ากระหายเลือด!"]
    },
    {
        "question": "คุณถูกเชิญให้เข้าร่วมภารกิจที่เคยทำ คุณมีความรู้สึกอยากเข้าร่วมแค่ไหน?",
        "options": ["ตื่นเต้นเต็มที่", "ค่อนข้างสนใจ", "เฉย ๆ", "ไม่ค่อยสนใจ", "ไม่สนใจเลย"]
    },
    {
        "question": "เมื่อสิ้นสุดวัน คุณรู้สึกกระวนกระวายเหมือนมีสิ่งที่ต้องทำมากมาย คุณรู้สึกถึงความไม่สงบในจิตใจแค่ไหน?",
        "options": ["สงบนิ่งเหมือนหิน", "กระวนกระวายนิดหน่อย", "กระวนกระวายปานกลาง", "เริ่มรู้สึกกระวนกระวาย", "ต้องเคลื่อนไหวอย่างไม่หยุด!"]
    }
]

feature_based_questions = [
    {
        "question": "คุณเดินอยู่บนเส้นทางที่เงียบสงบในยามพระอาทิตย์ตก คุณหยุดพักและคิดถึงชีวิตของคุณและทุกสิ่งที่คุณได้ผ่านมาจนถึงตอนนี้\n\nคุณรู้สึกพึงพอใจกับเส้นทางชีวิตของคุณแค่ไหน?",
        "options": ["ฉันแทบไม่เคยรู้สึกพึงพอใจเลย", "ฉันรู้สึกพึงพอใจบ้างแต่ยังสามารถมีความสุขได้มากกว่านี้", "โดยทั่วไปแล้วฉันรู้สึกพึงพอใจกับเส้นทางของฉัน", "ฉันรู้สึกภูมิใจในเส้นทางที่ฉันมาไกลได้ถึงขนาดนี้"]
    },
    {
        "question": "เส้นทางนำคุณไปยังลำธารที่มีน้ำไหลอย่างสงบ เมื่อคุณฟังเสียงน้ำไหล คุณสะท้อนถึงความสงบในสถานการณ์ที่กดดัน\n\nเมื่อเผชิญกับสถานการณ์ที่กดดัน คุณรู้สึกสงบแค่ไหน?",
        "options": ["ฉันแทบไม่เคยรู้สึกสงบเลย ความคิดของฉันยุ่งเหยิง", "ฉันจัดการความสงบได้บ้างแต่ยังรู้สึกกังวล", "ฉันส่วนใหญ่รู้สึกสงบแม้ในสถานการณ์ที่กดดัน", "ฉันรู้สึกสงบและมั่นคงเต็มที่"]
    },
    {
        "question": "เมื่อคุณเดินต่อไป คุณเจอทางแยก ทางหนึ่งดูเรียบง่าย อีกทางหนึ่งดูท้าทายแต่ก็น่าสนใจ คุณหยุดพักคิดถึงแรงจูงใจของคุณในการเดินทาง\n\nคุณรู้สึกมีแรงจูงใจในการเผชิญความท้าทายในชีวิตบ่อยแค่ไหน?",
        "options": ["ฉันมักจะขาดแรงจูงใจ", "ฉันมีแรงจูงใจอยู่บ้างแต่ไม่แน่นอน", "โดยทั่วไปแล้วฉันรู้สึกมีแรงจูงใจและพร้อม", "ฉันรู้สึกมีแรงจูงใจและตื่นเต้นกับความท้าทาย"]
    },
    {
        "question": "คุณผ่านทะเลสาบที่เงียบสงบซึ่งคุณสามารถมองเห็นภาพสะท้อนของตัวเองในน้ำ เมื่อคุณมองตัวเอง คุณเริ่มสะท้อนถึงการกระทำและการตัดสินใจในอดีตของคุณ\n\nคุณรู้สึกผิดหวังกับการตัดสินใจของตัวเองบ่อยแค่ไหน?",
        "options": ["ฉันมักจะรู้สึกผิดหวัง", "ฉันรู้สึกผิดหวังบางครั้ง", "ฉันแทบไม่เคยรู้สึกผิดหวัง ฉันยอมรับการตัดสินใจของตัวเอง", "ฉันแทบไม่เคยรู้สึกผิดหวัง ฉันมีความสงบในใจ"]
    },
    {
        "question": "เมฆเริ่มรวมตัวกัน สร้างเงาบางๆ บนเส้นทาง คุณรู้สึกเศร้าเมื่อคิดถึงความท้าทายในชีวิต\n\nเมื่อคิดถึงความยากลำบากในชีวิต คุณรู้สึกเศร้ามากแค่ไหน?",
        "options": ["ฉันมักจะรู้สึกเศร้าลึกๆ", "ฉันรู้สึกเศร้าแต่ก็สามารถจัดการได้ดี", "ฉันรู้สึกเศร้าบางครั้ง", "ฉันแทบไม่รู้สึกเศร้า ฉันมองไปยังความหวัง"]
    },
    {
        "question": "ข้างหน้า คุณเห็นเนินเขาที่มีเส้นทางชัน มันท้าทาย แต่คุณรู้ว่ามันจะคุ้มค่าถ้าถึงยอด\n\nคุณยอมรับปัญหาที่ท้าทายได้ดีแค่ไหน?",
        "options": ["ฉันมีความลำบากในการยอมรับ", "ฉันจัดการยอมรับบางปัญหาได้", "ฉันยอมรับปัญหาได้ดีเป็นส่วนใหญ่", "ฉันมองว่าความท้าทายเป็นโอกาส"]
    },
    {
        "question": "ขณะปีนขึ้น คุณพบอุปสรรคที่ทดสอบความอดทนและความแข็งแกร่งของคุณ คุณหยุดเพื่อประเมินว่าคุณจัดการกับอารมณ์ได้ดีแค่ไหน\n\nคุณสามารถควบคุมอารมณ์ของคุณในช่วงเวลาที่ยากลำบากได้ดีแค่ไหน?",
        "options": ["ฉันมักจะมีปัญหาในการสงบสติ", "ฉันจัดการได้แต่ก็พบว่าเป็นเรื่องยาก", "ฉันส่วนใหญ่รักษาความสงบและสมดุลได้", "ฉันรักษาความสงบและมีสมาธิอย่างสม่ำเสมอ"]
    },
    {
        "question": "เมื่อคุณขึ้นถึงยอดเขา คุณมองไปข้างหน้าและเห็นเส้นทางที่นำไปสู่จุดหมายที่ไม่รู้จัก คุณพิจารณาความมั่นใจของคุณในการเผชิญอนาคต\n\nคุณมั่นใจแค่ไหนในการเผชิญกับสิ่งที่ไม่แน่นอนในชีวิต?",
        "options": ["ฉันรู้สึกไม่แน่นอนและกังวล", "ฉันค่อนข้างมั่นใจแต่ยังลังเล", "ฉันรู้สึกมั่นใจในส่วนใหญ่", "ฉันรู้สึกมั่นใจเต็มที่และพร้อม"]
    },
    {
        "question": "ในระหว่างทาง คุณพบกับนักเดินทางที่ดูเหนื่อยล้า คุณรู้สึกถึงความเห็นอกเห็นใจต่อเขา\n\nคุณรู้สึกเห็นอกเห็นใจต่อผู้อื่นที่กำลังลำบากบ่อยแค่ไหน?",
        "options": ["ฉันแทบไม่เคยรู้สึกเห็นอกเห็นใจ", "ฉันรู้สึกเห็นอกเห็นใจบางครั้ง", "ฉันมักจะรู้สึกเห็นอกเห็นใจและห่วงใย", "ฉันรู้สึกเห็นอกเห็นใจอย่างลึกซึ้งต่อผู้ที่ต้องการ"]
    },
    {
        "question": "คุณเสนอให้นักเดินทางคนนั้นน้ำและรอยยิ้มที่เป็นมิตร คุณรู้สึกถึงความอบอุ่นภายในจากการช่วยเหลือผู้อื่น\n\nคุณรู้สึกยินดีมากแค่ไหนเมื่อช่วยเหลือผู้อื่น?",
        "options": ["ฉันแทบไม่รู้สึกยินดีในการช่วยเหลือ", "ฉันรู้สึกยินดีบางครั้งในการช่วยเหลือ", "ฉันรู้สึกยินดีอย่างมากเมื่อช่วยเหลือ", "การช่วยเหลือผู้อื่นทำให้ฉันรู้สึกเติมเต็ม"]
    },
    {
        "question": "เมื่อเดินต่อไป คุณคิดถึงความสำเร็จของคุณและความภูมิใจที่คุณรู้สึกต่อตัวเอง\n\nคุณรู้สึกภูมิใจต่อตัวเองมากแค่ไหน?",
        "options": ["ฉันแทบไม่เคยรู้สึกภูมิใจ", "ฉันรู้สึกภูมิใจเล็กน้อย", "ฉันรู้สึกภูมิใจต่อตัวเองเป็นส่วนใหญ่", "ฉันรู้สึกภูมิใจต่อตัวเองอย่างยิ่ง"]
    },
    {
        "question": "เส้นทางนำคุณกลับไปยังที่ที่คุ้นเคย ที่ที่คุณรู้สึกปลอดภัยและอบอุ่นเหมือนอยู่กับคนที่คุณรัก\n\nคุณรู้สึกปลอดภัยแค่ไหนในสภาพแวดล้อมครอบครัวของคุณ?",
        "options": ["ฉันแทบไม่เคยรู้สึกปลอดภัย", "ฉันรู้สึกปลอดภัยบางครั้ง", "ฉันรู้สึกปลอดภัยเป็นส่วนใหญ่", "ฉันรู้สึกปลอดภัยและได้รับการสนับสนุนอย่างสมบูรณ์"]
    },
    {
        "question": "คุณคิดถึงช่วงเวลาที่ยากลำบากและการสนับสนุนที่คุณได้รับจากครอบครัว\n\nคุณมั่นใจแค่ไหนว่าครอบครัวของคุณจะสนับสนุนคุณในเวลาที่ต้องการ?",
        "options": ["ฉันมีความมั่นใจน้อยในการสนับสนุนของครอบครัว", "ฉันค่อนข้างมั่นใจ", "ฉันรู้สึกมั่นใจในการสนับสนุนของพวกเขา", "ฉันมีความมั่นใจอย่างมากในการสนับสนุนของครอบครัว"]
    },
    {
        "question": "การเดินทางทำให้คุณรู้สึกซาบซึ้งถึงความสัมพันธ์ที่คุณมีร่วมกับครอบครัว\n\nคุณรู้สึกเชื่อมโยงกับสมาชิกในครอบครัวของคุณแค่ไหน?",
        "options": ["ฉันแทบไม่เคยรู้สึกเชื่อมโยง", "ฉันรู้สึกเชื่อมโยงบ้างเล็กน้อย", "ฉันรู้สึกใกล้ชิดและเชื่อมโยงกับครอบครัวมาก","ฉันรู้สึกใกล้ชิดและเชื่อมโยงกับครอบครัวมาก"]
    },
    {
    "question": "หลังจากการเดินทางอันยาวนาน คุณได้กลับมายังบ้านและรู้สึกถึงความอบอุ่นและความรักที่มาจากครอบครัว\n\nคุณรู้สึกถึงความรักและผูกพันระหว่างคุณและสมาชิกในครอบครัวของคุณแค่ไหน?",
    "options": ["ฉันแทบไม่เคยรู้สึกถึงความรักและผูกพัน", "ฉันรู้สึกถึงความรักและผูกพันบ้างบางครั้ง", "ฉันรู้สึกถึงความรักและผูกพันเป็นส่วนใหญ่", "ฉันรู้สึกถึงความรักและผูกพันอย่างลึกซึ้ง"]
    }
]

depression_questions = [
    {
        "question": "เช้าวันแรกในแดนมหัศจรรย์ อ้อดตื่นมาแล้วรู้สึกอย่างไร?",
        "options": ["ไม่อยากตื่นขึ้นเลย หมดกำลังใจ", "รู้สึกเบื่อเล็กน้อย", "มีพลังพร้อมเผชิญหน้า", "ตื่นเต้นเต็มที่กับการผจญภัย"]
    },
    {
        "question": "ขณะเดินชมวิวทุ่งดอกไม้ อ้อดรู้สึกอย่างไรกับธรรมชาติรอบตัว?",
        "options": ["ไม่มีอารมณ์สนใจสิ่งใดเลย", "รู้สึกดีขึ้นแต่คิดถึงเรื่องอื่นมากกว่า", "ชื่นชมธรรมชาติอย่างเต็มที่", "ตื่นเต้นที่ได้สัมผัสความงามตรงหน้า"]
    },
    {
        "question": "ระหว่างที่พบตัวละครอื่น ๆ ที่เข้ามาทักทาย อ้อดรู้สึกอย่างไร?",
        "options": ["ไม่อยากพูดคุยและหลีกเลี่ยง", "พยายามคุยแต่รู้สึกอึดอัด", "สนุกและอยากทำความรู้จัก", "เข้ากับคนอื่นได้ดี"]
    },
    {
        "question": "ตัวละครตัวหนึ่งชวนอ้อดไปสำรวจถ้ำที่ลึกลับ เขารู้สึกอย่างไร?",
        "options": ["ไม่อยากไปและปฏิเสธทันที", "รู้สึกไม่มั่นใจ", "อยากไปด้วยความตื่นเต้น", "ตัดสินใจไปโดยไม่ลังเล"]
    },
    {
        "question": "เมื่อเข้าไปในถ้ำอ้อดพบกับภาพวาดโบราณบนผนัง เขารู้สึกอย่างไร?",
        "options": ["รู้สึกเฉย ๆ และไม่สนใจ", "ชื่นชมเล็กน้อยแต่คิดถึงเรื่องอื่นมากกว่า", "รู้สึกตื่นเต้นและประทับใจ", "อยากศึกษาและจดจำรายละเอียดทั้งหมด"]
    },
    {
        "question": "ตัวละครลึกลับปรากฏตัวขึ้นและส่งข้อความแปลก ๆ ให้เขา อ้อดรู้สึกอย่างไร?",
        "options": ["รู้สึกกลัวและไม่สบายใจ", "ตกใจแต่พยายามทำใจดี ๆ", "มองว่าเป็นส่วนหนึ่งของการผจญภัย", "สนุกที่ได้พบความท้าทาย"]
    },
    {
        "question": "เมื่อมองกลับไปทางที่เดินผ่านมา อ้อดรู้สึกอย่างไรกับการเดินทางที่ผ่านมาทั้งหมด?",
        "options": ["รู้สึกว่ามันไร้ความหมายและเหนื่อยล้า", "รู้สึกเฉย ๆ ไม่ได้รู้สึกอะไรเป็นพิเศษ", "ภูมิใจที่มาถึงตรงนี้", "ตื่นเต้นที่จะไปต่อให้ถึงที่สุด"]
    },
    {
        "question": "เขาพบกับสะพานขาดที่ต้องหาทางข้ามไปอีกฝั่ง อ้อดรู้สึกอย่างไร?",
        "options": ["หมดกำลังใจและอยากถอย", "เครียดเล็กน้อยแต่พยายามหาทางแก้", "ตื่นเต้นที่จะหาทางข้าม", "เชื่อมั่นว่าจะข้ามได้"]
    },
    {
        "question": "เมื่อเห็นพระอาทิตย์ตกดินที่สวยงาม อ้อดรู้สึกอย่างไร?",
        "options": ["ไม่รู้สึกอะไรเป็นพิเศษ", "รู้สึกดีขึ้นเล็กน้อยแต่ไม่ได้ประทับใจมาก", "ตื่นเต้นและชื่นชมความงามอย่างเต็มที่", "รู้สึกสบายใจและผ่อนคลายมาก"]
    },
    {
        "question": "ขณะกำลังตั้งแคมป์ เขารู้สึกถึงความเหงาและคิดถึงบ้าน อ้อดรู้สึกอย่างไร?",
        "options": ["รู้สึกเหงาและเศร้าใจมาก", "รู้สึกเหงาแต่พยายามหากิจกรรมทำ", "คิดถึงคนที่รักแต่รู้สึกโอเค", "สนุกกับการผจญภัยและไม่ได้รู้สึกเหงา"]
    },
    {
        "question": "ตัวละครปริศนามาเสนอภารกิจพิเศษที่มีรางวัลใหญ่ อ้อดรู้สึกอย่างไร?",
        "options": ["ไม่อยากรับภารกิจและรู้สึกไม่มีพลัง", "รู้สึกลังเลเล็กน้อยแต่ก็ยอมรับ", "ตื่นเต้นและรับภารกิจทันที", "มองว่านี่คือความท้าทายที่สนุก"]
    },
    {
        "question": "ระหว่างเดินทางไปทำภารกิจพิเศษ เขารู้สึกเหนื่อยมาก อ้อดคิดอย่างไร?",
        "options": ["รู้สึกท้อแท้และอยากเลิกภารกิจ", "รู้สึกเหนื่อยแต่พยายามฮึดสู้", "คิดว่าเป็นเพียงอุปสรรคเล็กน้อย", "สนุกกับการเดินทางแม้จะเหนื่อย"]
    },
    {
        "question": "เขาได้รับข่าวดีเกี่ยวกับรางวัลที่ใหญ่กว่าที่คาด อ้อดรู้สึกอย่างไร?",
        "options": ["รู้สึกเฉย ๆ ไม่ตื่นเต้น", "ยิ้มเล็กน้อยแต่ไม่ได้ดีใจมาก", "รู้สึกดีใจและมีพลังเพิ่ม", "ดีใจและตื่นเต้นมากที่จะไปให้ถึงจุดหมาย"]
    },
    {
        "question": "เมื่อถึงจุดสูงสุดของการเดินทาง อ้อดพบว่ารางวัลไม่ใช่อย่างที่คิด เขารู้สึกอย่างไร?",
        "options": ["ผิดหวังมากและรู้สึกหมดกำลังใจ", "รู้สึกผิดหวังแต่ยอมรับได้", "คิดว่ายังมีโอกาสอีกมาก", "มองว่าเป็นโอกาสใหม่ในการเดินทางครั้งหน้า"]
    },
    {
        "question": "หลังการเดินทางที่เต็มไปด้วยเรื่องราว อ้อดรู้สึกอย่างไรเมื่อกลับถึงบ้าน?",
        "options": ["รู้สึกเศร้าที่การเดินทางจบลง", "รู้สึกคิดถึงการผจญภัยเล็กน้อย", "รู้สึกพอใจที่ได้เรียนรู้ประสบการณ์ใหม่", "พร้อมและตื่นเต้นที่จะเริ่มการผจญภัยใหม่"]
    }
]


@app.route('/download_report')
def download_report():
    # Define the PDF filename
    pdf_filename = 'user_report.pdf'
    document = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    # Get user responses from the session
    user_responses = session.get('responses', [])
    assessment = session.get('assessment', 'depression')  # Assuming depression assessment for example

    # Define feature names based on the assessment
    if assessment == 'depression':
        feature_names = [
            "Little_interest_pleasure", "Feeling_down_depressed", "Sleep_trouble",
            "Feeling_tired", "Appetite_issues", "Feeling_bad_about_self",
            "Trouble_concentrating", "Movement_changes", "Thoughts_of_self_harm",
            "Difficulty_making_decisions", "Feeling_worthless", "Physical_symptoms",
            "Social_withdrawal", "Irritability", "Lack_of_motivation"
        ]
    elif assessment == 'stress':
        feature_names = [
            "Difficulty_Sleeping", "Feeling_Overwhelmed", "Physical_Tension",
            "Difficulty_Concentrating", "Increased_Irritability", "Avoidance_Social_Situations",
            "Fatigue", "Change_in_Appetite", "Loss_of_Interest", "Feeling_Restless"
        ]
    else:
        feature_names = [f"Feature_{i + 1}" for i in range(len(user_responses))]

    # Ensure the length of feature names and user responses match
    if len(user_responses) != len(feature_names):
        feature_names = feature_names[:len(user_responses)]

    # Create styles regardless of data availability
    styles = getSampleStyleSheet()

    if not user_responses:
        # If no responses, handle gracefully
        elements.append(Paragraph("No data available to generate the report.", styles['Title']))
    else:
        # Create a title for the report
        title = Paragraph(f"{assessment.capitalize()} Assessment Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # User information and assessment details
        user_info = Paragraph("User: John Doe<br/>Date: 2024-11-03", styles['Normal'])
        elements.append(user_info)
        elements.append(Spacer(1, 12))

        # Prepare Data for Report Table
        data = [
            ['Assessment Factor', 'Your Score', 'Interpretation']
        ]

        for i, response in enumerate(user_responses):
            factor = feature_names[i]
            score = response
            # Convert score to percentage (assuming 1-5 scale)
            percentage = (score / 5) * 100
            interpretation = ""

            if percentage >= 80:
                interpretation = "High"
            elif 50 <= percentage < 80:
                interpretation = "Moderate"
            else:
                interpretation = "Low"

            data.append([factor, f"{score} / 5", f"{interpretation} ({percentage:.1f}%)"])

        # Create a Table with User Data
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 24))

        # Add Summary Paragraph
        summary = Paragraph(
            "Based on your responses, there are several areas to consider for improving your mental health. "
            "Please consult a healthcare professional for more personalized guidance.",
            styles['Normal']
        )
        elements.append(summary)
        elements.append(Spacer(1, 24))

        # Create a Bar Chart to visualize user responses
        drawing = Drawing(400, 200)
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        chart.data = [user_responses]

        chart.categoryAxis.categoryNames = feature_names
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.barWidth = 10
        chart.groupSpacing = 15
        chart.bars.strokeWidth = 0.5

        drawing.add(chart)
        elements.append(drawing)

    # Build the PDF
    document.build(elements)

    # Return the file as a downloadable response
    return send_file(pdf_filename, as_attachment=True)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_stress')
def start_stress():
    # Initialize session to store answers for stress assessment
    session['responses'] = []
    session['assessment'] = 'stress'
    return redirect(url_for('question', q_num=0))

@app.route('/start_anxiety')
def start_anxiety():
    # Initialize session to store answers for anxiety assessment
    session['responses'] = []
    session['assessment'] = 'feature_based'  # You might use 'anxiety' here if you have a separate model for anxiety
    return redirect(url_for('question', q_num=0))

@app.route('/start_bipolar')
def start_bipolar():
    session['responses'] = []
    session['assessment'] = 'depression'
    return redirect(url_for('question', q_num=0))

@app.route('/question/<int:q_num>', methods=['GET', 'POST'])
def question(q_num):
    # Define questions and images for each assessment
    assessment_data = {
        'stress': {
            'questions': stress_questions,
            'images': [f'stress_question_{i}.webp' for i in range(1, 11)]
        },
        'feature_based': {
            'questions': feature_based_questions,
            'images': [f'feature_question_{i}.png' for i in range(1, 16)]
        },
        'depression': {
            'questions': depression_questions,
            'images': [f'depression_question_{i}.png' for i in range(1, 16)]
        }
    }

    # Retrieve current assessment data from session
    assessment = session.get('assessment', 'stress')
    assessment_info = assessment_data.get(assessment, {'questions': [], 'images': []})

    questions = assessment_info['questions']
    images = assessment_info['images']

    # Check if we have reached the end of the questions
    if q_num >= len(questions):
        return redirect(url_for('result'))  # Redirect to result page after last question

    if request.method == 'POST':
        # Retrieve the selected answer and save it in the session
        answer = request.form.get('answer')
        responses = session.get('responses', [])
        responses.append(int(answer))
        session['responses'] = responses  # Save back to session
        return redirect(url_for('question', q_num=q_num + 1))  # Move to the next question

    # Shuffle options and store them in session
    options = questions[q_num]["options"][:]
    random.shuffle(options)
    session['current_options'] = options

    # Use url_for to generate the correct path to the image
    question_image_url = url_for('static', filename=f'images/{images[q_num]}')
    total_questions = len(questions)

    return render_template(
        'question.html',
        question=questions[q_num]["question"],
        q_num=q_num,
        options=options,
        question_image_url=question_image_url,
        total_questions=total_questions  # Pass the total number of questions here
    )


@app.route('/result')
def result():
    assessment = session.get('assessment', 'stress')
    responses = session.get('responses', [])

    if not responses:
        return render_template('result.html', message="No data available. Please complete the assessment.")

    # Initialize all variables with None
    stress_level = None
    stress_type = None
    Depression_Level = None
    result_output = None

    if assessment == 'stress':
        # For stress assessment
        feature_names = [
            "Difficulty_Sleeping", "Feeling_Overwhelmed", "Physical_Tension",
            "Difficulty_Concentrating", "Increased_Irritability", "Avoidance_Social_Situations",
            "Fatigue", "Change_in_Appetite", "Loss_of_Interest", "Feeling_Restless"
        ]
        input_data = pd.DataFrame([responses], columns=feature_names)

        try:
            predictions = stress_model.predict(input_data)
            stress_level, stress_type = predictions[0]
        except Exception as e:
            print(f"Error during stress model prediction: {e}")

    elif assessment == 'depression':
        # For depression assessment
        depression_feature_names = [
            "Little_interest_pleasure", "Feeling_down_depressed", "Sleep_trouble",
            "Feeling_tired", "Appetite_issues", "Feeling_bad_about_self",
            "Trouble_concentrating", "Movement_changes", "Thoughts_of_self_harm",
            "Difficulty_making_decisions", "Feeling_worthless", "Physical_symptoms",
            "Social_withdrawal", "Irritability", "Lack_of_motivation"
        ]
        input_data = pd.DataFrame([responses], columns=depression_feature_names)

        try:
            predictions = depression_model.predict(input_data)
            depression_score = predictions[0]
            if depression_score == 'Mild':
                Depression_Level = "Mild. You might be experiencing occasional low moods..."
            elif depression_score == 'Moderate':
                Depression_Level = "Moderate. You may be dealing with noticeable symptoms..."
            elif depression_score == 'Severe':
                Depression_Level = "Severe. It appears that you may be experiencing significant depressive symptoms..."
            else:
                Depression_Level = f"{depression_score}. Please consider taking actions to improve your mental health."
        except Exception as e:
            print(f"Error during depression model prediction: {e}")

    else:
        # For feature-based assessment
        input_data = pd.DataFrame([responses], columns=[f"Feature_{i + 1}" for i in range(len(responses))])

        try:
            predictions = overall_model.predict(input_data)
            score = predictions[0]
            result_output = interpret_mental_health_score(score)
        except Exception as e:
            print(f"Error during overall model prediction: {e}")

    return render_template(
        'result.html',
        stress_level=stress_level,
        stress_type=stress_type,
        Depression_Level=Depression_Level,
        result_output=result_output,
        download_link=url_for('download_report_pdf')  # Add download link for report
    )


# Only clear the session after the report is downloaded
@app.route('/download_report_pdf', methods=['POST'])
def download_report_pdf():
    # Get the user_name from the request form data
    user_name = request.form.get('user_name', 'Anonymous')  # Default to 'Anonymous' if not provided

    # Define the PDF filename
    pdf_filename = 'user_report.pdf'
    document = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    # Get user responses from the session
    user_responses = session.get('responses', [])
    assessment = session.get('assessment', 'depression')

    # Define feature names based on the assessment
    if assessment == 'depression':
        feature_names = [
            "Little_interest_pleasure", "Feeling_down_depressed", "Sleep_trouble",
            "Feeling_tired", "Appetite_issues", "Feeling_bad_about_self",
            "Trouble_concentrating", "Movement_changes", "Thoughts_of_self_harm",
            "Difficulty_making_decisions", "Feeling_worthless", "Physical_symptoms",
            "Social_withdrawal", "Irritability", "Lack_of_motivation"
        ]
    elif assessment == 'stress':
        feature_names = [
            "Difficulty_Sleeping", "Feeling_Overwhelmed", "Physical_Tension",
            "Difficulty_Concentrating", "Increased_Irritability", "Avoidance_Social_Situations",
            "Fatigue", "Change_in_Appetite", "Loss_of_Interest", "Feeling_Restless"
        ]
    else:
        feature_names = [f"Feature_{i + 1}" for i in range(len(user_responses))]

    # Truncate or extend feature_names to match the length of user_responses
    feature_names = feature_names[:len(user_responses)]

    # Create styles regardless of data availability
    styles = getSampleStyleSheet()

    if not user_responses:
        # If no responses, handle gracefully
        elements.append(Paragraph("No data available to generate the report.", styles['Title']))
    else:
        # Create a title for the report
        title = Paragraph(f"Assessment Report for {user_name}", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # User information and assessment details
        user_info = Paragraph(f"User: {user_name}<br/>Date: 2024-11-03", styles['Normal'])
        elements.append(user_info)
        elements.append(Spacer(1, 12))

        # Prepare Data for Report Table
        data = [
            ['Assessment Factor', 'Your Score', 'Interpretation']
        ]

        for i, response in enumerate(user_responses):
            factor = feature_names[i]
            score = response
            # Convert score to percentage (assuming 1-5 scale)
            percentage = (score / 5) * 100
            interpretation = "High" if percentage >= 80 else "Moderate" if percentage >= 50 else "Low"
            data.append([factor, f"{score} / 5", f"{interpretation} ({percentage:.1f}%)"])

        # Create a Table with User Data
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 24))
        drawing = Drawing(500, 300)
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 150
        chart.width = 400
        chart.data = [user_responses]

        # Customizing the chart to make it visually similar to the uploaded image
        chart.categoryAxis.categoryNames = feature_names
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.dy = -15
        chart.categoryAxis.labels.fontName = 'Helvetica'
        chart.categoryAxis.labels.fontSize = 8
        chart.barWidth = 15
        chart.groupSpacing = 10
        chart.bars[0].fillColor = colors.blue

        # Customizing the value axis (y-axis) to match the scale
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 5
        chart.valueAxis.valueStep = 1
        chart.valueAxis.labels.fontName = 'Helvetica'
        chart.valueAxis.labels.fontSize = 10

        # Customizing the bar border to give it a sharper look
        chart.bars.strokeWidth = 0.5

        # Add the chart to the drawing
        drawing.add(chart)
        elements.append(drawing)



    # Build the PDF
    document.build(elements)

    return send_file(pdf_filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
