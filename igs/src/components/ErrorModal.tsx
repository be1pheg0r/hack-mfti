import './ErrorModal.css'
// import blueFaceGif from '../assets/blue_face.gif'

import blueFaceGif from '../assets/hamster.gif'


type ErrorModalProps = {
  isOpen: boolean
  onClose: () => void
}

export function ErrorModal({ isOpen, onClose }: ErrorModalProps) {
  if (!isOpen) {
    return null
  }

  return (
    <div className="error-modal-overlay" role="dialog" aria-modal="true" aria-label="Окно с предупреждением об ошибках">
      <div className="error-modal-card">
        <p className="error-modal-message">Мирного пути не будет</p>
         <p>
            Ты много раз неправильно отвечал на вопросы. 
            Поэтому мы должны удалить твой компьютер.
            К сожалению, это единственный способ, 
            чтобы ты смог продолжить обучение и стать настоящим гением.
        </p>

         <p>
            Ладно, это шутка. Просто сфокусируйся и попробуй снова. У тебя все получится!
         </p>
        <div className="error-modal-media-row">
          <img
            className="error-modal-media"
            src={blueFaceGif}
            alt="Blue face"
          />
        </div>

        <button type="button" className="error-modal-close-button" onClick={onClose}>
          Продолжить
        </button>
      </div>
    </div>
  )
}
