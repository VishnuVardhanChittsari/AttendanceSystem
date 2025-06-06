from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from firebase_admin import firestore
from myproject.settings import db
from datetime import datetime

@csrf_exempt
def add_student(request):
    if request.method == 'POST':
        pin_number = request.POST.get('pin_number')
        name = request.POST.get('name')
        
        if pin_number and name:
            doc_ref = db.collection('students').document(pin_number)
            doc_ref.set({
                'name': name,
                'status': 'absent',
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            return JsonResponse({'status': 'success', 'message': 'Student added'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid data'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

# @csrf_exempt
# def add_attendance(request):
#     if request.method == 'POST':
#         pin_number = request.POST.get('pin_number')
#         status = request.POST.get('status')
        
#         if pin_number and status:
#             today_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
#             # Reference to the 'attendance' subcollection
#             attendance_ref = db.collection('students').document(pin_number).collection('attendance').document(today_date)
#             attendance_ref.set({
#                 'status': status,
#                 'timestamp': firestore.SERVER_TIMESTAMP
#             })
#             # Update latest status in the 'students' document
#             doc_ref = db.collection('students').document(pin_number)
#             doc_ref.update({
#                 'status': status,
#                 'timestamp': firestore.SERVER_TIMESTAMP
#             })
#             return JsonResponse({'status': 'success', 'message': 'Attendance recorded'})
#         else:
#             return JsonResponse({'status': 'error', 'message': 'Invalid data'})
#     else:
#         return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

# def get_student_data(request, pin_number):
#     student_doc = db.collection('students').document(pin_number).get()
#     if student_doc.exists:
#         student_data = student_doc.to_dict()
#         # Get historical attendance data
#         attendance_ref = db.collection('students').document(pin_number).collection('attendance').order_by('timestamp')
#         attendance_data = [doc.to_dict() for doc in attendance_ref.stream()]
#         return JsonResponse({
#             'student_info': student_data,
#             'attendance_records': attendance_data
#         })
#     else:
#         return JsonResponse({'status': 'error', 'message': 'Student not found'})


def index(request):
    return render(request, 'index.html')




# curl -X POST http://localhost:8000/attendance/add_student/ -d "pin_number=21551A0572" -d "name=Vishnu Vardhan"
# py manage.py runserver 192.168.29.52:8000 




# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status
# import firebase_admin
# from firebase_admin import credentials, firestore










from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase (Ensure this path points to your Firebase Admin SDK JSON file)


@api_view(['POST'])
def update_attendance(request):
    try:
        # Get attendance data from the request
        attendance_data = request.data

        # Check if data is a list of dictionaries
        if not isinstance(attendance_data, list):
            return Response({"error": "Invalid data format. Expected a list of dictionaries."}, status=status.HTTP_400_BAD_REQUEST)

        # Get the current date as a string
        today_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        # Store the attendance data in Firebase
        for student in attendance_data:
            pin_number = student.get('student_id')
            if not pin_number:
                return Response({"error": "Missing student_id in the data."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Reference to the 'attendance' subcollection for the specific student and date
            attendance_ref = db.collection('students').document(pin_number).collection('attendance').document(today_date)
            attendance_ref.set({
                'status': 'present',
                'timestamp': firestore.SERVER_TIMESTAMP,
            })
 
            # Update the latest status in the 'students' document
            doc_ref = db.collection('students').document(pin_number)
            doc_ref.update({
                'status': 'present',
                'timestamp': firestore.SERVER_TIMESTAMP
            })

        return Response({"message": "Attendance updated successfully"}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
























