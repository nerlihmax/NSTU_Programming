package ru.kheynov.cinemabooking.jwt.token

data class TokenClaim(
    val name: String,
    val value: String,
)