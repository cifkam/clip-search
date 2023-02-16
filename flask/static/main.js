var dragTimer;

function addInputFileDragListener(form)
{
    var element = form.querySelector(".input-file-upload");

    var borderDrag = '3px dashed crimson';
    var borderOld = element.style.border;

    $(document).on('dragover', function(e) {
        var dt = e.originalEvent.dataTransfer;
        if (dt.types && (dt.types.indexOf ? dt.types.indexOf('Files') != -1 : dt.types.contains('Files'))) {
          element.style.border = borderDrag;
          window.clearTimeout(dragTimer);
        }
    });
    $(document).on('dragleave', function(e) {
        dragTimer = window.setTimeout(function() {
            element.style.border = borderOld;
        }, 25);
    });
    
      

}