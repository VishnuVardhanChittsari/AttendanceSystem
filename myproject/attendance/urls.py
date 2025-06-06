from django.urls import path
from attendance import views

urlpatterns = [
    path('add_student/', views.add_student, name='add_student'),
    # path('add_attendance/', views.add_attendance, name='add_attendance'),
    # path('get_student/<str:pin_number>/', views.get_student_data, name='get_student_data'),
    path('index/', views.index, name='index'),
    # path('video_feed/', views.video_feed, name='video_feed'),
    path('api/update_attendance/', views.update_attendance, name='update_attendance'),
]



