<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AuthController;

Route::prefix('api')->group(function () {
    Route::post('/login', [AuthController::class, 'login']);
    Route::get('/checkAuth', [AuthController::class, 'checkAuth']);
});

Route::get('/{any}', function () {
    return view('welcome');
})->where('any', '.*');
