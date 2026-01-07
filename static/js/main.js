document.addEventListener('DOMContentLoaded', function() {
    var fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(function(input) {
        input.addEventListener('change', function(e) {
            var file = e.target.files[0];
            if (!file) return;
            
            var validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
            if (validTypes.indexOf(file.type) === -1) {
                alert('Invalid file type. Only PNG, JPG and JPEG are allowed.');
                e.target.value = '';
                return;
            }
            
            if (file.size > 10 * 1024 * 1024) {
                alert('File too large. Maximum size is 10MB.');
                e.target.value = '';
                return;
            }
        });
    });
    
    var forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function() {
            var submitBtn = form.querySelector('input[type="submit"], button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.value = 'Processing...';
                if (submitBtn.tagName === 'BUTTON') {
                    submitBtn.textContent = 'Processing...';
                }
            }
        });
    });
});
