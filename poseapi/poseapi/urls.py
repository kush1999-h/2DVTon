from django.contrib import admin
from django.urls import path
from poseapp.views import process_frame

urlpatterns = [
    path('admin/', admin.site.urls),
    path('process_frame/', process_frame),
]
