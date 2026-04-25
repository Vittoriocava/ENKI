from django.urls import path
from . import views

app_name = "enki_dashboard"

urlpatterns = [
    path("", views.map_view, name="map"),
    path("api/floods/", views.flood_data, name="flood_data"),
    path("api/route/",  views.route,      name="route"),
]
