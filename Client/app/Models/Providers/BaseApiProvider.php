<?php

namespace App\Models\Providers;

use Illuminate\Support\Facades\Http;

class BaseApiProvider {
    private function get($url) {
        return Http::get($url);
    }

    private function post($url, $data) {
        return Http::post($url, $data);
    }

    private function put($url, $data) {
        return Http::put($url, $data);
    }

    private function delete($url, $data) {
        return Http::delete($url, $data);
    }
}
