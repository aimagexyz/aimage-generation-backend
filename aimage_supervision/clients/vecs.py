from vecs import client as vecs_client


class VecsClient(vecs_client.Client):
    def __init__(self, connection_string: str):
        '''
        Initialize a Client instance.

        Args:
            connection_string (str): A string representing the database connection information.

        Returns:
            None
        '''
        # Initialize the SQLAlchemy engine and metadata.
        db_url = connection_string.replace('postgres://', 'postgresql://')
        self.engine = vecs_client.create_engine(
            url=db_url,
            pool_size=10,
            max_overflow=2,
            pool_recycle=300,
            pool_pre_ping=True,
            pool_use_lifo=True,
        )
        self.meta = vecs_client.MetaData(schema='vecs')
        self.Session = vecs_client.sessionmaker(self.engine)

        with self.Session() as session:
            with session.begin():
                session.execute(vecs_client.text(
                    'create schema if not exists vecs;',
                ))
                session.execute(vecs_client.text(
                    'create extension if not exists vector;',
                ))
                self.vector_version: str = session.execute(vecs_client.text(
                    "select installed_version from pg_available_extensions where name = 'vector' limit 1;",
                )).scalar_one()


def create_vecs_client(connection_string: str) -> VecsClient:
    '''Creates a client from a Postgres connection string'''
    return VecsClient(connection_string)
