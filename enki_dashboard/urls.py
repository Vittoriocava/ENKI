from django.urls import path
from . import views

app_name = "enki_dashboard"

urlpatterns = [
    path("", views.map_view, name="map"),
    path("api/floods/", views.flood_data,  name="flood_data"),
    path("api/route/",  views.route,       name="route"),
    path("api/spawn/",  views.spawn_flood, name="spawn_flood"),
    # path("api/clear-spawns/", views.clear_spawns, name="clear_spawns"),
]
