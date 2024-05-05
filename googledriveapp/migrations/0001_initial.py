# Generated by Django 5.0.3 on 2024-04-15 04:01

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Folder",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("foldername", models.CharField(max_length=50)),
                ("folderdesc", models.CharField(max_length=255)),
                (
                    "folderuser",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="File",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("filetitle", models.CharField(max_length=50)),
                ("file", models.FileField(upload_to="Files")),
                (
                    "year",
                    models.CharField(
                        choices=[
                            ("Year 1", "Year 1"),
                            ("Year 2", "Year 2"),
                            ("Year 3", "Year 3"),
                            ("Year 4", "Year 4"),
                            ("Year 5", "Year 5"),
                            ("Year 6", "Year 6"),
                            ("General (N/A)", "General (N/A)"),
                        ],
                        default="General (N/A)",
                        max_length=20,
                    ),
                ),
                ("course", models.CharField(max_length=100)),
                ("course_unit", models.CharField(max_length=100)),
                ("description", models.TextField(blank=True, null=True)),
                ("namespace", models.CharField(max_length=255, unique=True)),
                (
                    "likes",
                    models.ManyToManyField(
                        blank=True,
                        related_name="liked_files",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "folder",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="files",
                        to="googledriveapp.folder",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Bookmark",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="bookmarks",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "file",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="bookmarks",
                        to="googledriveapp.file",
                    ),
                ),
                (
                    "folder",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="bookmarks",
                        to="googledriveapp.folder",
                    ),
                ),
            ],
        ),
    ]