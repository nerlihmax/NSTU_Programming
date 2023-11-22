package ru.kheynov.hotel.jwt.token

data class TokenClaim(
    val name: String,
    val value: String,
)