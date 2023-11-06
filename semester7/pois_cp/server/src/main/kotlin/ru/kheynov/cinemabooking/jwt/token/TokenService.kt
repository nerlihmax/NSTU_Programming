package ru.kheynov.cinemabooking.jwt.token

interface TokenService {
    fun generateTokenPair(config: TokenConfig, vararg claims: TokenClaim): TokenPair
}