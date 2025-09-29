<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AuthController;
use App\Http\Controllers;

Route::prefix('api')->group(function () {
    Route::post('/login', [AuthController::class, 'login']);
    Route::get('/checkAuth', [AuthController::class, 'checkAuth']);
    Route::post('/template', [Controllers\Constructor::class, 'createTemplate']);
    Route::get('/template', [Controllers\Constructor::class, 'getTemplates']);
    Route::delete('/template', [Controllers\Constructor::class, 'deleteTemplate']);
    Route::post('/element', [Controllers\Constructor::class, 'createElement']);
    Route::get('/token', function () {
        return csrf_token();
    });
});

Route::get('/{any}', function () {
    return view('welcome');
})->where('any', '.*');
