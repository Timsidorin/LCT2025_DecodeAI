<?php

namespace App\Models\Providers;

use Illuminate\Http\Client\ConnectionException;
use Illuminate\Support\Facades\Http;

class BaseApiProvider
{
    private $baseUrl;

    public function __construct($baseUrl)
    {
        $this->baseUrl = $baseUrl;
    }

    protected function get(string $url, array $headers = [])
    {
        return Http::withHeaders($headers)->get($this->baseUrl . $url);
    }

    /**
     * @throws ConnectionException
     */
    protected function post(string $url, array $data, bool $isFormData = false, array $headers = [])
    {
        $http = Http::withHeaders($headers);

        if ($isFormData) {
            return $http->asForm()->post($this->baseUrl . $url, $data);
        } else {
            return $http->post($this->baseUrl . $url, $data);
        }
    }

    protected function put(string $url, array $data, array $headers = [])
    {
        return Http::withHeaders($headers)->put($this->baseUrl . $url, $data);
    }

    protected function delete(string $url, array $data, array $headers = [])
    {
        return Http::withHeaders($headers)->delete($this->baseUrl . $url, $data);
    }
}
