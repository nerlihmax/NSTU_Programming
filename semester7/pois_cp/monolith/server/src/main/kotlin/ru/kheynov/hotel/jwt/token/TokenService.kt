package ru.kheynov.hotel.jwt.token

interface TokenService {
    fun generateTokenPair(config: TokenConfig, vararg claims: TokenClaim): TokenPair
}