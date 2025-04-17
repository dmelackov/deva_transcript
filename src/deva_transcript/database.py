
from typing import Annotated

from faststream import Depends
from config import settings
from deva_p1_db.database import DatabaseSessionManager, get_db_url
from sqlalchemy.ext.asyncio import AsyncSession

session_manager = DatabaseSessionManager(get_db_url(settings.db_user,
                                                    settings.db_password,
                                                    settings.db_ip,
                                                    settings.db_port,
                                                    settings.db_name),
                                         {"echo": False})

Session = Annotated[AsyncSession, Depends(session_manager.session)]