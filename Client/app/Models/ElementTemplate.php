<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ElementTemplate extends Model
{

    protected $fillable = [
        'data',
        'template_id'
    ];

    public function template()
    {
        return $this->belongsTo(Template::class);
    }
}
